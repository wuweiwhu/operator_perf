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
TILE_M = K
TILE_N = Q
STAGE_A = 4
STAGE_B = 4
MULTICAST = 8
CLUSTER_COUNTS = 3
SM_COUNTS = 24
SM_MMA_MACS = 4096
MMA_UTIL = 0.84 * 56 / 64
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
DDR_UTIL = 0.70
FORCE_HIT = True
PROLOGUE_CYCLES_EXTRA = 3000
EPILOGUE_CYCLES_EXTRA = 0

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
            return 58 * TILE_C * 2
        elif data_type == "C":
            return TILE_M * TILE_N * OUTPUT_ROWS_PER_CTA * 2
        else:
            raise ValueError("Unknown data type")

    def access(self, data_type, start_X, start_Y, start_Z = 0):
        self.access_count += 1
        if FORCE_HIT:
            return True, 0
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
    def __init__(self, cache, id = 0):
        self.cache = cache
        self.cta_id = id
    def bind(self, tile_m, tile_n):
        self.tile_m = tile_m
        self.tile_n = tile_n
        #self.tma_a_cycles = [0 for _ in range(STAGE_A)]
        #self.tma_b_cycles = [0 for _ in range(STAGE_B)]
        self.tma_cycles = [0 for _ in range(STAGE_A)]
        self.mma_cycles = [0 for _ in range(STAGE_A)]
        # for detecting if stage b steped
        self.last_stage_b = -1

    def execute(self, ifeature_row_iter, r_iter, s_iter, c_iter, stage_a, stage_b):
        if self.done():
            return
        # start of output row
        coord_start_output_row = self.tile_n * OUTPUT_ROWS_PER_CTA * STRIDE_H
        coord_start_input_row = coord_start_output_row + ifeature_row_iter * STRIDE_H - PAD_H

        if stage_b != self.last_stage_b:
            #print(f"CTA{self.cta_id} Load ifeature row:{coord_start_input_row} Channel:{c_iter}")
            B_hit, evict = L2.access("B", coord_start_input_row, c_iter, 0)
            B_L2C_Transfer_Bytes_Per_SM = L2.sizeof("B")
            B_NOC_Transfer_Bytes_Per_SM = B_L2C_Transfer_Bytes_Per_SM
            B_DDR_Transfer_Bytes_Per_SM = 0
            if not B_hit:
                B_DDR_Transfer_Bytes_Per_SM = L2.sizeof("B") + evict
            self.last_stage_b = stage_b
        else:
            B_hit = True
            B_L2C_Transfer_Bytes_Per_SM = 0
            B_DDR_Transfer_Bytes_Per_SM = 0
            B_NOC_Transfer_Bytes_Per_SM = 0

        #print(f"CTA{self.cta_id} Load filter R{r_iter} S{s_iter} C{c_iter}")
        A_hit, evict = L2.access("A", r_iter, s_iter, c_iter)
        A_L2C_Transfer_Bytes_Per_SM = L2.sizeof("A") / MULTICAST
        A_NOC_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM * MULTICAST
        A_DDR_Transfer_Bytes_Per_SM = 0
        if not A_hit:
            A_DDR_Transfer_Bytes_Per_SM = (L2.sizeof("A") + evict) / MULTICAST


        L2C_Transfer_Bytes_Per_SM = A_L2C_Transfer_Bytes_Per_SM + B_L2C_Transfer_Bytes_Per_SM
        NOC_Transfer_Bytes_Per_SM = A_NOC_Transfer_Bytes_Per_SM + B_NOC_Transfer_Bytes_Per_SM
        DDR_Transfer_Bytes_Per_SM = A_DDR_Transfer_Bytes_Per_SM + B_DDR_Transfer_Bytes_Per_SM
        Serilization_Cycles = max(L2C_Transfer_Bytes_Per_SM / (L2_RD_BW_PER_SM / (STAGE_A-1)), NOC_Transfer_Bytes_Per_SM / (NOC_RD_BW_PER_SM / (STAGE_A-1)), DDR_Transfer_Bytes_Per_SM / (DDR_BW_PER_SM / (STAGE_A-1)))

        if A_hit and B_hit:
            TMA_Cycles = Serilization_Cycles + L2_RT_LAT
        else:
            TMA_Cycles = Serilization_Cycles + DDR_RT_LAT

        MMA_Cycles = TILE_M * TILE_N * TILE_C / (SM_MMA_MACS * MMA_UTIL)
        #if self.cta_id == 0:
            #print(f"CTA{self.cta_id} MMA Start Cycle{max(self.mma_cycles)}")
        self.tma_cycles[stage_a % STAGE_A] = self.mma_cycles[stage_a % STAGE_A] + MBARRIER_SYNC_CYCLES + TMA_Cycles
        mma_idle_cycles = 0 if stage_a == 0 else self.mma_cycles[(stage_a - 1)%STAGE_A]
        self.mma_cycles[stage_a % STAGE_A] = max(self.tma_cycles[stage_a % STAGE_A], mma_idle_cycles) + MBARRIER_SYNC_CYCLES + MMA_Cycles
    def done(self):
        return self.tile_m == None or self.tile_n == None
    def cycles(self):
        if self.done():
            return 0
        # FIXME
        coord_start_m = self.tile_m
        coord_start_n = self.tile_n
        _, evict = L2.access("C", coord_start_m, coord_start_n, 0)
        C_Cycles = max(L2.sizeof("C") / (L2_WR_BW_PER_SM * L2_UTIL) + L2_RT_LAT, evict / (DDR_BW_PER_SM * DDR_UTIL) + (DDR_RT_LAT - L2_RT_LAT))
        #ADMEM read cycles should be overlapped
        mainloop_cycles = max(max(self.tma_cycles), max(self.mma_cycles))
        if self.cta_id == 0:
            print(f"prologue: {PROLOGUE_CYCLES_EXTRA}, mainloop:{mainloop_cycles} cycles, epilogue:{C_Cycles + EPILOGUE_CYCLES_EXTRA} cycles")
        Tile_Cycles = C_Cycles + MBARRIER_SYNC_CYCLES + mainloop_cycles + PROLOGUE_CYCLES_EXTRA+ EPILOGUE_CYCLES_EXTRA
        return Tile_Cycles

def get_cta_tasks():
    tile_m = 0
    tile_n = 0
    for tile_m in range(K // TILE_M):
        for tile_n in range(SM_COUNTS):
            yield (tile_m, tile_n)
    while(True):
        yield (None, None)


task_generator = get_cta_tasks()
total_tile_cycles = 0
ctas = [CTA(L2, id) for id in range(SM_COUNTS)]


while(True):
    all_done = True
    for cta in ctas:
        (tile_m, tile_n) = next(task_generator)
        cta.bind(tile_m, tile_n)
        all_done = all_done and cta.done()

    if all_done:
        break
    
    stage_a = 0
    stage_b = 0
    # mainloop
    for ifeature_row_iter in range(OUTPUT_ROWS_PER_CTA + R - 1):
        coord_start_r = max(0, ifeature_row_iter - (OUTPUT_ROWS_PER_CTA - 1))
        coord_end_r = min(R, ifeature_row_iter + 1)
        for c_iter in range(C//TILE_C):
            for s_iter in range(S):
                for r_iter in range(coord_start_r, coord_end_r):
                    for cta in ctas:
                        cta.execute(ifeature_row_iter, r_iter, s_iter, c_iter, stage_a, stage_b)
                    stage_a += 1
            stage_b += 1
                    
    for cta in ctas:
        #print(cta.cycles())
        total_tile_cycles += cta.cycles()

total_cycles = total_tile_cycles / SM_COUNTS
print(f"Total cycles: {total_cycles}")
print(f"MMA Utilization: {N * P * Q * R * S * C * K / (SM_MMA_MACS * SM_COUNTS) / total_cycles * 100}%")
