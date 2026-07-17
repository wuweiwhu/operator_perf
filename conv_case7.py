import math
N = 1
H = 56
W = 56
P = 56
Q = 56
R = 3
S = 3
C = 128
K = 128
PAD_H = 1
PAD_W = 1
Stride=1
Dilation=1
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

    def access(self, data_type, start_X, start_Y, start_Z):
        self.access_count += 1
        if (data_type, start_X, start_Y, start_Z) in self.cache:
            self.cache[(data_type, start_X, start_Y, start_Z)] = self.access_count
            self.hit_count += 1
            return True, 0
        else:
            if self.occupancy + self.sizeof(data_type) <= self.size:
                self.cache[(data_type, start_X, start_Y, start_Z)] = self.access_count
                self.occupancy += self.sizeof(data_type)
                return False, 0
            else:
                # evict the least recently used block
                lru_key = min(self.cache, key=self.cache.get)
                del self.cache[lru_key]
                self.cache[(data_type, start_X, start_Y, start_Z)] = self.access_count
                if lru_key[0] == "C":
                    return False, self.sizeof("C")
                else:
                    return False, 0

L2 = L2CACHE(size=36 * 1024 * 1024 * 0.90)  # 36MB L2 cache, effective size 90% of total


def get_cta_tasks(): #persisten kernel
    output_rows_per_cta_floor = math.floor(N * P / SM_COUNTS)
    output_rows_per_cta_ceil = math.ceil( N * P / SM_COUNTS)
    cta_counts_more_task = N * P % SM_COUNTS
    for task_id in range(SM_COUNTS):
        if task_id < SM_COUNTS - cta_counts_more_task:
            yield task_id * output_rows_per_cta_floor
        else:
            yield (task_id - (SM_COUNTS - cta_counts_more_task))*output_rows_per_cta_ceil + (SM_COUNTS - cta_counts_more_task)*output_rows_per_cta_floor

STAGE_A = 4
STAGE_B = 4

for output_row in get_cta_tasks():
    Tile_TMA_B_Cycles = [0 for _ in range(STAGE_A)]
    Tile_TMA_A_Cycles = [0 for _ in range(STAGE_B)]
    for r in range(R):
        start_w = -PAD_W
        start_h = r-PAD_H
        for start_c in range(C):
            B_L2C_Transfer_Bytes_Per_SM = L2.sizeof("B") / 8
            B_NOC_Transfer_Bytes_Per_SM = L2.sizeof("B") / 4
            B_DDR_Transfer_Bytes_Per_SM = 0
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
            for s in range(S):
                L2.access("A", start_w, start_h, start_c)


    
