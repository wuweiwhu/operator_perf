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
STRIDE_H=1
STRIDE_W=1
DILATION=1
OUTPUT_ROWS_PER_CTA = 3
TILE_C = 64
TILE_M = 128
TILE_N = 58
K_STAGE = 4
MULTICAST = 8
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
            return TILE_M * TILE_C * 2
        elif data_type == "B":
            return TILE_N * TILE_C * 2
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

class CTA:
    def __init__(self, cache):
        self.cache = cache
    def bind(self, tile_m, tile_n):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tma_a_cycles = [0 for _ in range(K_STAGE)]
        self.tma_b_cycles = [0 for _ in range(K_STAGE)]
        self.mma_cycles = [0 for _ in range(K_STAGE)]
    
    def execute(self, ifeature_row_iter, r_iter, s_iter, c_iter, load_ifeature):
        if self.done():
            return
        # start of output row
        coord_start_output_row = self.tile_n * OUTPUT_ROWS_PER_CTA * STRIDE_H
        coord_start_input_row = coord_start_output_row + ifeature_row_iter * STRIDE_H - PAD_H

        if load_ifeature:
            B_hit, evict = L2.access("B", coord_start_input_row, c_iter, 0)
            B_L2C_Transfer_Bytes_Per_SM = L2.sizeof("B")
            B_NOC_Transfer_Bytes_Per_SM = B_L2C_Transfer_Bytes_Per_SM
            B_DDR_Transfer_Bytes_Per_SM = 0
            if not B_hit:
                B_DDR_Transfer_Bytes_Per_SM = L2.sizeof("B") + evict

        A_hit, evict = L2.access("A", r_iter, s_iter, c_iter)
        A_L2C_Transfer_Bytes_Per_SM = L2.sizeof("A") / MULTICAST
        A_NOC_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM * MULTICAST
        A_DDR_Transfer_Bytes_Per_SM = 0
        if not A_hit:
            A_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("A") + evict) / MULTICAST

        if load_ifeature:
            L2C_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM + B_L2C_Transfer_Bytes_Per_SM
            NOC_Transfer_Bytes_Per_SM = A_NOC_Transfer_Bytes_Per_SM + B_NOC_Transfer_Bytes_Per_SM
            DDR_Transfer_Bytes_Per_SM = A_DDR_Transfer_Bytes_Per_SM + B_DDR_Transfer_Bytes_Per_SM
        else:
            L2C_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM
            NOC_Transfer_Bytes_Per_SM = A_NOC_Transfer_Bytes_Per_SM
            DDR_Transfer_Bytes_Per_SM = A_DDR_Transfer_Bytes_Per_SM

        Serilization_Cycles = max(L2C_Transfer_Bytes_Per_SM / (L2_RD_BW_PER_SM / (K_STAGE-1)), NOC_Transfer_Bytes_Per_SM / (NOC_RD_BW_PER_SM / (K_STAGE-1)), DDR_Transfer_Bytes_Per_SM / (DDR_BW_PER_SM / (K_STAGE-1)))
        if A_hit and B_hit:
            TMA_Cycles = Serilization_Cycles + L2_RT_LAT
        else:
            TMA_Cycles = Serilization_Cycles + DDR_RT_LAT

        MMA_Cycles = TILE_M * TILE_N * TILE_C / (SM_MMA_MACS * MMA_UTIL)

        self.tma_a_cycles[tile_k % K_STAGE] = self.mma_cycles[tile_k % K_STAGE] + MBARRIER_SYNC_CYCLES + TMA_Cycles
        self.tma_b_cycles[]
        
        #mma_idle_cycles = 0
        #for stage in range(1, min(K_STAGE, tile_k+1)):
            #mma_idle_cycles = max(self.mma_cycles[(tile_k-stage) % K_STAGE], mma_idle_cycles)
        #self.mma_cycles[tile_k % K_STAGE] = max(self.tma_cycles[tile_k % K_STAGE] + MBARRIER_SYNC_CYCLES, mma_idle_cycles) + MMA_Cycles
    def done(self):
        return self.tile_m == None or self.tile_n == None
    def cycles(self):
        if self.done():
            return 0
        return 0

def get_cta_tasks(): #persisten kernel
    tile_m = 0
    tile_n = 0
    for tile_m in range(K // TILE_M):
        for tile_n in range(SM_COUNTS):
            yield (tile_m, tile_n)
    while(True):
        yield (None, None)


task_generator = get_cta_tasks()
total_tile_cycles = 0
ctas = [CTA(L2) for _ in range(SM_COUNTS)]


while(True):
    all_done = True
    for cta in ctas:
        (tile_m, tile_n) = next(task_generator)
        cta.bind(tile_m, tile_n)
        all_done = all_done and cta.done()

    if all_done:
        break
    
    for ifeature_row_iter in range(OUTPUT_ROWS_PER_CTA + R - 1):
        coord_start_r = max(0, ifeature_row_iter - (OUTPUT_ROWS_PER_CTA - 1))
        coord_end_r = min(R, ifeature_row_iter + 1)
        for c_iter in range(C//TILE_C):
            load_ifeature = True
            for s_iter in range(S):
                for r_iter in range(coord_start_r, coord_end_r):
                    for cta in ctas:
                        cta.execute(ifeature_row_iter, r_iter, s_iter, c_iter, load_ifeature)
                        #print(ifeature_row_iter, r_iter, s_iter, c_iter)
                    load_ifeature = False
    for cta in ctas:
        print(cta.cycles())
        total_tile_cycles += cta.cycles()