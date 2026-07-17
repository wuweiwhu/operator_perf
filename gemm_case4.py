import math
Prob_M = 4096
Prob_N = 4096
Prob_K = 4096
BLOCKS_IN_GGA = 8
TILE_M_CGA = 512
TILE_N_CGA = 512
TILE_K = 64
CLUSTER_COUNTS = 3
SM_COUNTS = 24
SM_MMA_MACS = 4096
MMA_UTIL = 0.83
MBARRIER_SYNC_CYCLES = 20 #?
L2_RT_LAT = 250
L2_RD_BW_PER_SM = 96
L2_WR_BW_PER_SM = 48
L2_UTIL = 0.85
NOC_RD_BW_PER_SM = 128
NOC_WR_BW_PER_SM = 64
NOC_UTIL = 0.85
DDR_RT_LAT = 850
DDR_BW_PER_SM = 32
DDR_UTIL = 0.70

class L2CACHE:
    def __init__(self, size):
        self.size = size
        self.occupancy = 0
        self.cache = dict()
        self.hit_count = 0
        self.access_count = 0
    
    def sizeof(self, data_type):
        if data_type == "A":
            return 512 * 64 * 2
        elif data_type == "B":
            return 512 * 64 * 2
        elif data_type == "C":
            return 512 * 512 * 2
        else:
            raise ValueError("Unknown data type")

    def access(self, data_type, start_X, start_Y):
        self.access_count += 1
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
    for task_id in range(Prob_M // TILE_M_CGA * Prob_N // TILE_N_CGA):
        old_x = task_id // 8
        old_y = task_id %8
        tile_x = ((old_y & 1) << 2) + ((old_x >> 2) << 1) + (old_x & 1)
        tile_y = ((old_y >> 1) << 1) + ((old_x >> 1) & 1)
        yield (tile_x, tile_y)
      
Total_Cycles = 0
K_STAGE = 6

CGA_TILES = Prob_M // TILE_M_CGA * Prob_N // TILE_N_CGA

# FIXME: consider parallelism
for (tile_x, tile_y) in get_cga_tasks(): 
    Coord_start_M = tile_y * TILE_M_CGA
    Coord_start_N = tile_x * TILE_N_CGA
    #print(f"Processing Coord ({Coord_start_M}, {Coord_start_N})")
    Tile_MMA_Cycles = [0 for _ in range(K_STAGE)]
    Tile_TMA_Cycles = [0 for _ in range(K_STAGE)]
    Tile_Cycles = 0
    for Tile_K in range(Prob_K // TILE_K):
        Coord_start_K = Tile_K * TILE_K
        A_L2C_Transfer_Bytes_Per_SM = L2.sizeof("A") / 8
        A_NOC_Transfer_Bytes_Per_SM = L2.sizeof("A") / 4
        A_DDR_Transfer_Bytes_Per_SM = 0
        B_L2C_Transfer_Bytes_Per_SM = L2.sizeof("B") / 8
        B_NOC_Transfer_Bytes_Per_SM = L2.sizeof("B") / 4
        B_DDR_Transfer_Bytes_Per_SM = 0
        A_hit, evict = L2.access("A", Coord_start_M, Coord_start_K)
        if not A_hit:
            A_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("A") + evict) / 8
        B_hit, evict = L2.access("B", Coord_start_N, Coord_start_K)
        if not B_hit:
            B_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("B") + evict) / 8
        L2C_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM + B_L2C_Transfer_Bytes_Per_SM
        NOC_Transfer_Bytes_Per_SM = A_NOC_Transfer_Bytes_Per_SM + B_NOC_Transfer_Bytes_Per_SM
        DDR_Transfer_Bytes_Per_SM = A_DDR_Transfer_Bytes_Per_SM + B_DDR_Transfer_Bytes_Per_SM
        Serilization_Cycles = max(L2C_Transfer_Bytes_Per_SM / (L2_RD_BW_PER_SM / (K_STAGE-1)), NOC_Transfer_Bytes_Per_SM / (NOC_RD_BW_PER_SM / (K_STAGE-1)), DDR_Transfer_Bytes_Per_SM / (DDR_BW_PER_SM / (K_STAGE-1)))
        if A_hit and B_hit:
            TMA_Cycles = Serilization_Cycles + L2_RT_LAT
        else:
            TMA_Cycles = Serilization_Cycles + DDR_RT_LAT

        MMA_Cycles = TILE_M_CGA * TILE_M_CGA * TILE_K / (SM_MMA_MACS * BLOCKS_IN_GGA * MMA_UTIL)

        Tile_TMA_Cycles[Tile_K % K_STAGE] = Tile_MMA_Cycles[Tile_K % K_STAGE] + MBARRIER_SYNC_CYCLES + TMA_Cycles
        MMA_Curr_Cycles = 0
        for stage in range(1, min(K_STAGE, Tile_K+1)):
            MMA_Curr_Cycles = max(Tile_MMA_Cycles[(Tile_K-stage) % K_STAGE], MMA_Curr_Cycles)
        Tile_MMA_Cycles[Tile_K % K_STAGE] = max(Tile_TMA_Cycles[Tile_K % K_STAGE] + MBARRIER_SYNC_CYCLES, MMA_Curr_Cycles) + MMA_Cycles

    C_hit, evict = L2.access("C", Coord_start_M, Coord_start_N)
    C_Cycles = max(L2.sizeof("C") / 8 / (L2_WR_BW_PER_SM * L2_UTIL) + L2_RT_LAT / 2, evict / 8 / (DDR_BW_PER_SM * DDR_UTIL) + (DDR_RT_LAT - L2_RT_LAT))
    TMA_Tile_Cycles = max(Tile_TMA_Cycles)
    MMA_Tile_Cycles = max(Tile_MMA_Cycles)
    Tile_Cycles = C_Cycles + MBARRIER_SYNC_CYCLES + max(TMA_Tile_Cycles, MMA_Tile_Cycles)
    #print(f"TMA Tile Cycles: {TMA_Tile_Cycles}, MMA Tile Cycle: {MMA_Tile_Cycles}")
    Total_Cycles += Tile_Cycles

Wave_Count = math.ceil(CGA_TILES / CLUSTER_COUNTS)
Total_Cycles = Total_Cycles / CGA_TILES * Wave_Count
print(f"Total cycles: {Total_Cycles}")
print(f"MMA Utilization: {Prob_M * Prob_N * Prob_K / (SM_MMA_MACS * SM_COUNTS) / Total_Cycles * 100}%")