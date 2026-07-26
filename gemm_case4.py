import math
ABTYPE = "NVF4"
CDTYPE = "F16"
#DTYPE = "NVF4"
Prob_M = 4096
Prob_N = 4096
Prob_K = 4096
#Prob_M = 2048
#Prob_N = 2048
#Prob_K = 11008
BLOCKS_IN_GGA = 8
MULTICAST_A = 2
MULTICAST_B = 2
K_STAGE = 4
TILE_M_CGA = 512
TILE_N_CGA = 512
TILE_K = 256
CLUSTER_COUNTS = 3
SM_COUNTS = 24
SM_MMA_MACS = 4096 * 8
MMA_UTIL = 0.92
MBARRIER_SYNC_CYCLES = 40
L2_RT_LAT = 270
L2_RD_BW_PER_SM = 96
L2_WR_BW_PER_SM = 48
L2_UTIL = 0.85
NOC_RD_BW_PER_SM = 128
NOC_WR_BW_PER_SM = 64
NOC_UTIL = 0.85
DDR_RT_LAT = 850
DDR_BW_PER_SM = 32
DDR_UTIL = min(0.70, 224*0.8*3/(DDR_RT_LAT - L2_RT_LAT))
FORCE_HIT = False
STREAMING_STORE = False
WRAM_UP = True
PROLOGUE_CYCLES_EXTRA = 0000
EPILOGUE_CYCLES_EXTRA = 0000

class CGA:
    def __init__(self, cache, id = 0):
        self.cache = cache
        self.cga_id = id
        self.clock = 0
    def bind(self, tile_m, tile_n):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tma_cycles = [0 for _ in range(K_STAGE)]
        self.mma_cycles = [0 for _ in range(K_STAGE)]
    def execute(self, tile_k):
        if self.done():
            return
        coord_start_m = self.tile_m
        coord_start_n = self.tile_n
        #print(f"Processing Coord ({coord_start_m}, {coord_start_n})")
        coord_start_k = tile_k
        A_L2C_Transfer_Bytes_Per_SM = L2.sizeof("A") / BLOCKS_IN_GGA
        A_NOC_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM * MULTICAST_A
        A_DDR_Transfer_Bytes_Per_SM = 0
        B_L2C_Transfer_Bytes_Per_SM = L2.sizeof("B") / BLOCKS_IN_GGA
        B_NOC_Transfer_Bytes_Per_SM = B_L2C_Transfer_Bytes_Per_SM * MULTICAST_B
        B_DDR_Transfer_Bytes_Per_SM = 0
        A_hit, evict = L2.access("A", coord_start_m, coord_start_k)
        if not A_hit:
            A_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("A") + evict) / BLOCKS_IN_GGA
        B_hit, evict = L2.access("B", coord_start_n, coord_start_k)
        if not B_hit:
            B_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("B") + evict) / BLOCKS_IN_GGA
        L2C_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM + B_L2C_Transfer_Bytes_Per_SM
        NOC_Transfer_Bytes_Per_SM = A_NOC_Transfer_Bytes_Per_SM + B_NOC_Transfer_Bytes_Per_SM
        DDR_Transfer_Bytes_Per_SM = A_DDR_Transfer_Bytes_Per_SM + B_DDR_Transfer_Bytes_Per_SM

        Serilization_Cycles = max(
            L2C_Transfer_Bytes_Per_SM / (L2_RD_BW_PER_SM * L2_UTIL), 
            NOC_Transfer_Bytes_Per_SM / (NOC_RD_BW_PER_SM * NOC_UTIL), 
            DDR_Transfer_Bytes_Per_SM / (DDR_BW_PER_SM * DDR_UTIL)
        )
        RT_LAT = L2_RT_LAT if (A_hit and B_hit) else DDR_RT_LAT

        request_issue_time = self.mma_cycles[tile_k % K_STAGE] + MBARRIER_SYNC_CYCLES
        data_ready_time = request_issue_time + RT_LAT
        bus_acquire_time = data_ready_time if (tile_k == 0) else max(data_ready_time, self.tma_cycles[(tile_k - 1) % K_STAGE])
        self.tma_cycles[tile_k % K_STAGE] = bus_acquire_time + Serilization_Cycles

        MMA_Cycles = TILE_M_CGA * TILE_N_CGA * TILE_K / (SM_MMA_MACS * BLOCKS_IN_GGA * MMA_UTIL)
        mma_idle_cycles = 0 if tile_k == 0 else self.mma_cycles[(tile_k - 1)%K_STAGE]
        self.mma_cycles[tile_k % K_STAGE] = max(self.tma_cycles[tile_k % K_STAGE], mma_idle_cycles) + MBARRIER_SYNC_CYCLES + MMA_Cycles
        if self.cga_id == 0:
            print(f"stage {tile_k} TMA: {self.tma_cycles[tile_k % K_STAGE]}, MMA: {self.mma_cycles[tile_k % K_STAGE]}")
    def done(self):
        return self.tile_m == None or self.tile_n == None
    def cycles(self):
        if self.done():
            return 0
        coord_start_m = self.tile_m
        coord_start_n = self.tile_n
        _, evict = L2.access("C", coord_start_m, coord_start_n)
        if not STREAMING_STORE:
            if evict > 0:
                evict_cycles = evict / BLOCKS_IN_GGA / (DDR_BW_PER_SM * DDR_UTIL) + (DDR_RT_LAT - L2_RT_LAT)
            else:
                evict_cycles = 0
            C_Cycles = max(L2.sizeof("C") / BLOCKS_IN_GGA / (L2_WR_BW_PER_SM * L2_UTIL) + L2_RT_LAT, evict_cycles)
        else:
            C_Cycles = L2.sizeof("C") / BLOCKS_IN_GGA / (DDR_BW_PER_SM * DDR_UTIL)
        mainloop_cycles = max(max(self.tma_cycles), max(self.mma_cycles))
        #if self.cga_id == 0:
            #print(f"prologue: {PROLOGUE_CYCLES_EXTRA}, mainloop:{mainloop_cycles} cycles, epilogue:{C_Cycles + EPILOGUE_CYCLES_EXTRA} cycles")
        Tile_Cycles = C_Cycles + MBARRIER_SYNC_CYCLES + mainloop_cycles + PROLOGUE_CYCLES_EXTRA+ EPILOGUE_CYCLES_EXTRA
        return Tile_Cycles


class L2CACHE:
    def __init__(self, size):
        self.size = size
        self.occupancy = 0
        # cache: {(data_type, start_X, start_Y): [last_access_timestamp, size_in_bytes]}
        self.cache = dict()
        self.hit_count = 0
        self.access_count = 0
        self.evict_class = {
            "C": 0,
            "A": 1,
            "B": 1
        }

    def sizeof(self, data_type, tile_m=TILE_M_CGA, tile_n=TILE_N_CGA, tile_k=TILE_K):
        dtype = {
            "NVF4": 1/2 * (1 + 1/8),  # 0.5B payload + 1/8 scale factor overhead
            "F16": 2,
            "F32": 4,
            "MXF8": 1 * (1 + 1/32)
        }
        if data_type == "A":
            return tile_m * tile_k * dtype[ABTYPE]
        elif data_type == "B":
            return tile_n * tile_k * dtype[ABTYPE]
        elif data_type == "C":
            return tile_m * tile_n * dtype[CDTYPE]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def access(self, data_type, start_X, start_Y, tile_m=TILE_M_CGA, tile_n=TILE_N_CGA, tile_k=TILE_K):
        self.access_count += 1
        key = (data_type, start_X, start_Y)
        req_size = self.sizeof(data_type, tile_m, tile_n, tile_k)

        if FORCE_HIT:
            self.hit_count += 1
            return True, 0.0
        if STREAMING_STORE and data_type == "C":
            return False, 0.0

        if key in self.cache:
            self.cache[key][0] = self.access_count
            self.hit_count += 1
            return True, 0.0

        evicted_dirty_bytes = 0.0
        while self.occupancy + req_size > self.size and len(self.cache) > 0:
            lru_key = min(self.cache.keys(), key=lambda k: (self.evict_class[k[0]], self.cache[k][0]))
            _, evicted_size = self.cache[lru_key]
            self.occupancy -= evicted_size
            if lru_key[0] == "C":
                evicted_dirty_bytes += evicted_size
            del self.cache[lru_key]

        self.cache[key] = [self.access_count, req_size]
        self.occupancy += req_size

        return False, evicted_dirty_bytes

L2 = L2CACHE(size=36 * 1024 * 1024 * 0.90)

if WRAM_UP:
    for x in range(Prob_M // TILE_M_CGA):
        for k in range(Prob_K // TILE_K):
            L2.access("A", x, k)
    for y in range(Prob_N // TILE_N_CGA):
        for k in range(Prob_K // TILE_K):
            L2.access("B", y, k)
    L2.hit_count = 0
    L2.access_count = 0

def get_cga_tasks():
    for old_y in range(Prob_N // TILE_N_CGA):
        for old_x in range(Prob_M // TILE_M_CGA):
            # FOR 4K * 4K
            tile_x = (old_x & 1) + ((old_x >> 2) << 1) + ((old_y & 1) << 2)
            tile_y = ((old_x >> 1) & 1) + (((old_y >> 1) & 1) << 1) + ((old_y >> 2) << 2)
            #tile_x = (old_x & 1) + ((old_x >> 2) << 1) + ((old_y >> 1) << 2)
            #tile_y = ((old_x >> 1) & 1) + ((old_y & 1) << 1) + ((old_y >> 2) << 2)
            # FOR 2K * 2K
            #tile_x = (old_x & 1) +  + ((old_y & 1) << 1)
            #tile_y = ((old_x >> 1) & 1) + ((old_y >> 1) << 1)
            yield (tile_x, tile_y)
    while(True):
        yield(None, None)

task_generator = get_cga_tasks()
total_tile_cycles = 0
total_cycles = 0
clusters = [CGA(L2, id) for id in range(CLUSTER_COUNTS)]
while(True):
    all_done = True
    for cluster in clusters:
        (tile_m, tile_n) = next(task_generator)
        cluster.bind(tile_m, tile_n)
        all_done = all_done and cluster.done()
    if all_done:
        break
    for tile_k in range(Prob_K // TILE_K):
        for cluster in clusters:
            cluster.execute(tile_k)
    for cluster in clusters:
        if not cluster.done():
            cycles_this_tile = cluster.cycles()
            print(f"Execute: {cluster.tile_m} {cluster.tile_n}, Cycles: {cycles_this_tile}")
            cluster.clock += cycles_this_tile
            total_cycles = max(cluster.clock, total_cycles)
            total_tile_cycles += cycles_this_tile

CGA_TILES = Prob_M // TILE_M_CGA * Prob_N // TILE_N_CGA
Wave_Count = math.ceil(CGA_TILES / CLUSTER_COUNTS)
total_avg_cycles = total_tile_cycles / CGA_TILES * Wave_Count
print(f"Total Cycles: {total_cycles}")
print(f"Total Avg Cycles: {total_avg_cycles}")
#print(f"Tile Hit Rate: {L2.hit_count / max(1, L2.access_count) * 100:.2f}% (Hits: {L2.hit_count} / Total: {L2.access_count})")
print(f"MMA Utilization: {Prob_M * Prob_N * Prob_K / (SM_MMA_MACS * SM_COUNTS) / total_cycles * 100}%")