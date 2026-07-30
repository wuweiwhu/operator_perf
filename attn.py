import math

# ================= FA4 GQA 模型参数 =================
DTYPE_BYTES = 1 # FP16
SEQ_Q = 4096
SEQ_KV = 4096
HEAD_DIM = 128

TILE_Q_M = 128
TILE_KV_N = 128
Q_TILES_PER_CTA = 2 

CLUSTER_COUNTS = 3
SM_COUNTS = 24
BLOCKS_IN_GGA = 8

# GQA 配置: 32 Q Heads, 4 KV Heads (8个CTA共享同一组KV)
MULTICAST_KV = 8 

K_STAGE = 4
SM_MMA_MACS = 16384 * 8 # 假设较高算力 
MMA_UTIL = 0.92
MBARRIER_SYNC_CYCLES = 40

# L2 & NOC 带宽配置
L2_RT_LAT = 270
L2_RD_BW_PER_SM = 96
L2_WR_BW_PER_SM = 48
L2_UTIL = 0.85
NOC_RD_BW_PER_SM = (96 + 128) / 2
NOC_WR_BW_PER_SM = 48
NOC_UTIL = 0.85
DDR_RT_LAT = 920
DDR_BW_PER_SM = 32
DDR_UTIL = min(0.70, 224*0.8*3/(DDR_RT_LAT - L2_RT_LAT))

SOFTMAX_CYCLES_PER_TILE = 140 

PROLOGUE_CYCLES_EXTRA = 3000
EPILOGUE_EXPOSED_RATIO = 0
EPILOGUE_CYCLES_EXTRA = 200
STREAMING_STORE = False
FORCE_HIT = False

# ================= L2 Cache 状态机 =================
class L2CACHE:
    def __init__(self, size):
        self.size = size
        self.occupancy = 0
        self.cache = dict()
        self.hit_count = 0
        self.stat_requests = 0
        self.access_count = 0
        
        # 优先级定义: O(输出)优先被踢, K/V(流式高复用)居中, Q(整个循环都驻留SRAM/L2)极力保护
        self.evict_class = {
            "O": 0,
            "K": 1,
            "V": 1,
            "Q": 2
        }

    def sizeof(self):
        return 128 * 128 * DTYPE_BYTES

    def access(self, data_type, cta_id, step_idx):
        self.access_count += 1
        self.stat_requests += 1
        key = (data_type, cta_id, step_idx)
        req_size = self.sizeof()

        if FORCE_HIT:
            self.hit_count += 1
            return True, 0.0
        if STREAMING_STORE and data_type == "O":
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
            if lru_key[0] == "O":
                evicted_dirty_bytes += evicted_size
            del self.cache[lru_key]

        self.cache[key] = [self.access_count, req_size]
        self.occupancy += req_size

        return False, evicted_dirty_bytes

L2 = L2CACHE(size=36 * 1024 * 1024 * 0.90)

# ================= Attention CTA 建模 =================
class AttentionCTA:
    def __init__(self, cta_id):
        self.cta_id = cta_id
        self.clock = PROLOGUE_CYCLES_EXTRA
        self.tc_clock = 0
        self.simt_clock = 0
        self.prev_s0_finish = 0
        self.prev_s1_finish = 0
        self.pending_epilogue_bus_cycles = 0
        self.pending_O_Cycles = 0

    def bind(self, q_block_idx):
        self.q_block_idx = q_block_idx
        self.tma_cycles = [0 for _ in range(K_STAGE)]
        self.tc_clock = 0
        self.simt_clock = 0
        self.prev_s0_finish = 0
        self.prev_s1_finish = 0
        self.mainloop_req_bus_cycles = 0
        
        # Prologue: L2 拉取 Q Tile
        q_ddr_bytes = 0
        for i in range(Q_TILES_PER_CTA):
            q_idx = q_block_idx * Q_TILES_PER_CTA + i
            # Q 是每个 CTA 独立拥有的，所以加入 cta_id 区分
            q_hit, q_evict = L2.access("Q", self.cta_id, q_idx)
            if not q_hit:
                q_ddr_bytes += (L2.sizeof() + q_evict) / BLOCKS_IN_GGA
        
        # 初始 Q 的 DDR 延迟计入绝对挂钟
        if q_ddr_bytes > 0:
            self.clock += q_ddr_bytes / (DDR_BW_PER_SM * DDR_UTIL) + DDR_RT_LAT

    def execute(self, kv_step):
        # 1. 精确的 L2 状态交互 (GQA 组播核心逻辑)
        # KV 是共享的，因此 cta_id 固定为 0，代表 8个CTA 去 L2 捞同一把 Key
        k_hit, k_evict = L2.access("K", 0, kv_step)
        v_hit, v_evict = L2.access("V", 0, kv_step)

        k_ddr_bytes = 0 if k_hit else (L2.sizeof() + k_evict) / BLOCKS_IN_GGA
        v_ddr_bytes = 0 if v_hit else (L2.sizeof() + v_evict) / BLOCKS_IN_GGA

        L2C_Transfer_Bytes = 2 * L2.sizeof() / BLOCKS_IN_GGA
        NOC_Transfer_Bytes = L2C_Transfer_Bytes * MULTICAST_KV
        DDR_Transfer_Bytes = k_ddr_bytes + v_ddr_bytes
        
        self.mainloop_req_bus_cycles += (L2C_Transfer_Bytes // 128) * 64 / (NOC_WR_BW_PER_SM * NOC_UTIL)

        Serilization_Cycles = max(
            L2C_Transfer_Bytes / (L2_RD_BW_PER_SM * L2_UTIL), 
            NOC_Transfer_Bytes / (NOC_RD_BW_PER_SM * NOC_UTIL), 
            DDR_Transfer_Bytes / (DDR_BW_PER_SM * DDR_UTIL)
        )
        
        RT_LAT = L2_RT_LAT if (k_hit and v_hit) else DDR_RT_LAT 

        request_issue_time = max(self.tc_clock, self.simt_clock) + MBARRIER_SYNC_CYCLES
        data_ready_time = request_issue_time + RT_LAT
        bus_acquire_time = data_ready_time if (kv_step == 0) else max(data_ready_time, self.tma_cycles[(kv_step - 1) % K_STAGE])
        
        self.tma_cycles[kv_step % K_STAGE] = bus_acquire_time + Serilization_Cycles
        kv_ready_time = self.tma_cycles[kv_step % K_STAGE]

        # 2. FA4 Ping-Pong 硬件流水线
        MACs_PER_TILE = TILE_Q_M * TILE_KV_N * HEAD_DIM
        MMA_CYCLES = MACs_PER_TILE / (SM_MMA_MACS / SM_COUNTS * MMA_UTIL)

        if kv_step > 0:
            pv0_start = max(self.tc_clock, self.prev_s0_finish)
            self.tc_clock = pv0_start + MMA_CYCLES
            
        qk0_start = max(self.tc_clock, kv_ready_time)
        qk0_finish = qk0_start + MMA_CYCLES
        self.tc_clock = qk0_finish
        
        s0_start = max(self.simt_clock, qk0_finish)
        s0_finish = s0_start + SOFTMAX_CYCLES_PER_TILE
        self.simt_clock = s0_finish
        
        if kv_step > 0:
            pv1_start = max(self.tc_clock, self.prev_s1_finish)
            self.tc_clock = pv1_start + MMA_CYCLES
            
        qk1_start = max(self.tc_clock, kv_ready_time)
        qk1_finish = qk1_start + MMA_CYCLES
        self.tc_clock = qk1_finish
        
        s1_start = max(self.simt_clock, qk1_finish)
        s1_finish = s1_start + SOFTMAX_CYCLES_PER_TILE
        self.simt_clock = s1_finish
        
        self.prev_s0_finish = s0_finish
        self.prev_s1_finish = s1_finish

    def cycles(self):
        mainloop_cycles = max(max(self.tma_cycles), self.tc_clock, self.simt_clock)

        # 排空最后一次 PV
        MACs_PER_TILE = TILE_Q_M * TILE_KV_N * HEAD_DIM
        MMA_CYCLES = MACs_PER_TILE / (SM_MMA_MACS / SM_COUNTS * MMA_UTIL)
        mainloop_cycles += 2 * MMA_CYCLES

        available_bus_cycles = mainloop_cycles - self.mainloop_req_bus_cycles
        required_epilogue_bus_cycles = self.pending_epilogue_bus_cycles + EPILOGUE_CYCLES_EXTRA
        
        if available_bus_cycles >= required_epilogue_bus_cycles or required_epilogue_bus_cycles == 0:
            epilogue_exposed_cycles = 0
        else:
            unhidden_bus_cycles = required_epilogue_bus_cycles - available_bus_cycles
            epilogue_exposed_cycles = (unhidden_bus_cycles / required_epilogue_bus_cycles) * self.pending_O_Cycles

        # O 矩阵写出与 L2 状态交互
        o_ddr_bytes = 0
        for i in range(Q_TILES_PER_CTA):
            q_idx = self.q_block_idx * Q_TILES_PER_CTA + i
            o_hit, o_evict = L2.access("O", self.cta_id, q_idx)
            if not STREAMING_STORE:
                o_ddr_bytes += (L2.sizeof() + o_evict) / BLOCKS_IN_GGA
            else:
                o_ddr_bytes += L2.sizeof() / BLOCKS_IN_GGA

        C_Cycles = o_ddr_bytes / (DDR_BW_PER_SM * DDR_UTIL) + DDR_RT_LAT

        self.pending_O_Cycles = C_Cycles
        self.pending_epilogue_bus_cycles = (2 * L2.sizeof()) / (NOC_WR_BW_PER_SM * NOC_UTIL)

        if EPILOGUE_EXPOSED_RATIO == 0:
            Tile_Cycles = epilogue_exposed_cycles + MBARRIER_SYNC_CYCLES + mainloop_cycles
        else:
            Tile_Cycles = self.pending_O_Cycles * EPILOGUE_EXPOSED_RATIO + MBARRIER_SYNC_CYCLES + mainloop_cycles + EPILOGUE_CYCLES_EXTRA
            
        return Tile_Cycles

# ================= 模拟执行主循环 =================
TOTAL_Q_BLOCKS_PER_CTA = SEQ_Q // (TILE_Q_M * Q_TILES_PER_CTA)
TOTAL_KV_STEPS = SEQ_KV // TILE_KV_N

total_cycles = 0
# 单个 Cluster 内 8 个 CTA
ctas = [AttentionCTA(id) for id in range(BLOCKS_IN_GGA)]

for q_block in range(TOTAL_Q_BLOCKS_PER_CTA):
    # 1. Cluster 内所有 CTA 同步 Bind 
    for cta in ctas:
        cta.bind(q_block)
        
    # 2.模拟 GQA 下的硬件同步执行
    # CTA0 会触发 KV 缺失进入 DDR，随后 CTA1-7 全部命中 L2！
    for kv_step in range(TOTAL_KV_STEPS):
        for cta in ctas:
            cta.execute(kv_step)
            
    # 3. 结算本轮 Cycles
    for cta in ctas:
        cycles_this_tile = cta.cycles()
        cta.clock += cycles_this_tile
        total_cycles = max(cta.clock, total_cycles)

# 流水线排空 (Pipeline Drain)
for cta in ctas:
    if cta.pending_O_Cycles > 0:
        final_epilogue_cycles = cta.pending_O_Cycles + EPILOGUE_CYCLES_EXTRA
        cta.clock += final_epilogue_cycles
        total_cycles = max(cta.clock, total_cycles)
        cta.pending_O_Cycles = 0 

print(f"Total Cycles: {total_cycles:.1f}")
print(f"L2 Hit Rate: {L2.hit_count / max(1, L2.stat_requests) * 100:.2f}% (Hits: {L2.hit_count} / Total: {L2.stat_requests})")

TOTAL_MACS = BLOCKS_IN_GGA * TOTAL_Q_BLOCKS_PER_CTA * TOTAL_KV_STEPS * (2 * TILE_Q_M * TILE_KV_N * HEAD_DIM) * 2
PEAK_MACS = total_cycles * (SM_MMA_MACS / SM_COUNTS * BLOCKS_IN_GGA * MMA_UTIL)
print(f"MMA Utilization: {TOTAL_MACS / PEAK_MACS * 100:.2f}%")