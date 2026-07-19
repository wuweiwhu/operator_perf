import math
Prob_M = 128
Prob_N = 2048
Prob_K = 2048
BLOCKS_IN_GGA = 8
MULTICAST = 8
K_STAGE = 4
TILE_M_CGA = 128
TILE_N_CGA = 1024
TILE_K = 128
CLUSTER_COUNTS = 2
SM_COUNTS = 24
SM_MMA_MACS = 16384
MMA_UTIL = 0.74
MBARRIER_SYNC_CYCLES = 20 #?
L2_RT_LAT = 270
# Only 16 SM, Should be full B/W
L2_RD_BW_PER_SM = 128
L2_WR_BW_PER_SM = 64
L2_UTIL = 0.85
NOC_RD_BW_PER_SM = 128
NOC_WR_BW_PER_SM = 64
NOC_UTIL = 0.85
DDR_RT_LAT = 850
DDR_BW_PER_SM = 32
DDR_UTIL = 0.70
FORCE_HIT = False

class CGA:
    def __init__(self, cache, id):
        self.cache = cache
        self.cga_id = id
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
        coord_start_k = tile_k * TILE_K
        A_L2C_Transfer_Bytes_Per_SM = L2.sizeof("A") / BLOCKS_IN_GGA
        A_NOC_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM * MULTICAST
        A_DDR_Transfer_Bytes_Per_SM = 0
        B_L2C_Transfer_Bytes_Per_SM = L2.sizeof("B") / BLOCKS_IN_GGA
        B_NOC_Transfer_Bytes_Per_SM = L2.sizeof("B") / BLOCKS_IN_GGA
        B_DDR_Transfer_Bytes_Per_SM = 0
        A_hit, evict = L2.access("A", coord_start_m, coord_start_k)
        if not A_hit:
            A_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("A") + evict) / 8
        B_hit, evict = L2.access("B", coord_start_n, coord_start_k)
        if not B_hit:
            B_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("B") + evict)
        L2C_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM + B_L2C_Transfer_Bytes_Per_SM
        NOC_Transfer_Bytes_Per_SM = A_NOC_Transfer_Bytes_Per_SM + B_NOC_Transfer_Bytes_Per_SM
        DDR_Transfer_Bytes_Per_SM = A_DDR_Transfer_Bytes_Per_SM + B_DDR_Transfer_Bytes_Per_SM
        Serilization_Cycles = max(L2C_Transfer_Bytes_Per_SM / (L2_RD_BW_PER_SM / (K_STAGE-1)), NOC_Transfer_Bytes_Per_SM / (NOC_RD_BW_PER_SM / (K_STAGE-1)), DDR_Transfer_Bytes_Per_SM / (DDR_BW_PER_SM / (K_STAGE-1)))
        if A_hit and B_hit:
            TMA_Cycles = Serilization_Cycles + L2_RT_LAT
        else:
            TMA_Cycles = Serilization_Cycles + DDR_RT_LAT

        MMA_Cycles = TILE_M_CGA * TILE_M_CGA * TILE_K / (SM_MMA_MACS * BLOCKS_IN_GGA * MMA_UTIL)

        self.tma_cycles[tile_k % K_STAGE] = self.mma_cycles[tile_k % K_STAGE] + MBARRIER_SYNC_CYCLES + TMA_Cycles
        #mma_idle_cycles = 0
        #for stage in range(1, min(K_STAGE, tile_k+1)):
        #    mma_idle_cycles = max(self.mma_cycles[(tile_k-stage) % K_STAGE], mma_idle_cycles)
        mma_idle_cycles = 0 if tile_k == 0 else self.mma_cycles[(tile_k - 1)%K_STAGE]
        self.mma_cycles[tile_k % K_STAGE] = max(self.tma_cycles[tile_k % K_STAGE] + MBARRIER_SYNC_CYCLES, mma_idle_cycles) + MMA_Cycles
    def done(self):
        return self.tile_m == None or self.tile_n == None
    def cycles(self):
        if self.done():
            return 0
        coord_start_m = self.tile_m * TILE_M_CGA
        coord_start_n = self.tile_n * TILE_N_CGA
        _, evict = L2.access("C", coord_start_m, coord_start_n)
        C_Cycles = max(L2.sizeof("C") / 8 / (L2_WR_BW_PER_SM * L2_UTIL) + L2_RT_LAT / 2, evict / 8 / (DDR_BW_PER_SM * DDR_UTIL) + (DDR_RT_LAT - L2_RT_LAT))
        TMA_Tile_Cycles = max(self.tma_cycles)
        MMA_Tile_Cycles = max(self.mma_cycles)
        Tile_Cycles = C_Cycles + MBARRIER_SYNC_CYCLES + max(TMA_Tile_Cycles, MMA_Tile_Cycles)
        return Tile_Cycles

class L2CACHE:
    def __init__(self, size):
        self.size = size
        self.occupancy = 0
        self.cache = dict()
        self.hit_count = 0
        self.access_count = 0
    
    def sizeof(self, data_type):
        if data_type == "A":
            return TILE_M_CGA * TILE_K * (1+1/32)
        elif data_type == "B":
            return TILE_N_CGA * TILE_K * (1+1/32)
        elif data_type == "C":
            return TILE_M_CGA * TILE_N_CGA * 2
        else:
            raise ValueError("Unknown data type")

    def access(self, data_type, start_X, start_Y):
        self.access_count += 1
        if FORCE_HIT:
            return True, 0
        if (data_type, start_X, start_Y) in self.cache:
            self.cache[(data_type, start_X, start_Y)] = self.access_count
            self.hit_count += 1
            return True, 0
        else:
            if self.occupancy + self.sizeof(data_type) <= self.size:
                self.cache[(data_type, start_X, start_Y)] = self.access_count
                self.occupancy += self.sizeof(data_type)
                return False, 0
            else:
                # evict the least recently used block
                lru_key = min(self.cache, key=self.cache.get)
                del self.cache[lru_key]
                self.cache[(data_type, start_X, start_Y)] = self.access_count
                if lru_key[0] == "C":
                    return False, self.sizeof("C")
                else:
                    return False, 0

L2 = L2CACHE(size=36 * 1024 * 1024 * 0.90)  # 36MB L2 cache, effective size 90% of total

def get_cga_tasks():
    for tile_x in range(Prob_M // TILE_M_CGA):
        for tile_y in range(Prob_N // TILE_N_CGA):
            yield (tile_x, tile_y)
    while(True):
        yield(None, None)

task_generator = get_cga_tasks()
total_tile_cycles = 0
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
        #print(cluster.cycles())
        total_tile_cycles += cluster.cycles()

CGA_TILES = Prob_M // TILE_M_CGA * Prob_N // TILE_N_CGA
Wave_Count = math.ceil(CGA_TILES / CLUSTER_COUNTS)
total_cycles = total_tile_cycles / CGA_TILES * Wave_Count
print(f"Total cycles: {total_cycles}")
print(f"MMA Utilization: {Prob_M * Prob_N * Prob_K / (SM_MMA_MACS * SM_COUNTS) / total_cycles * 100}%")