#!/usr/bin/env python3
"""
🧠 SYNTHETIC CAUSAL AGENT v450.0 (DYNAMIC MULTI-COMBINATION GENERATOR)
- Автономно генерирует все типы когнитивных комбинаций:
    • Single: Одиночный изолированный концепт (K=1.0)
    • Flat List: Равноположенные сиблинги без вложенности (K=1.0)
    • Nesting A ⊃ B: Замок НА Горе (K=2.5, положительный сдвиг iPLV)
    • Inverted B ⊃ A: Гора В Замке (K=2.5, отрицательный сдвиг iPLV)
    • Deep 3-Level: Космос ⊃ Планета ⊃ Киберпанк (K=3.8)
- В калибровке строго держит запрашиваемый слот учителя CLIP.
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 500.0
CHUNK_SIZE = 20  # 40 мс при 500 Гц
NUM_CHANNELS = 16
NUM_DEVICES = 4
TWO_PI = 2.0 * math.pi

COORDS_X = np.array([10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
                     -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
                      2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71], dtype=np.float32)
COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0))

ALL_CONCEPT_NAMES = ["КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ", "ГОРА", "ЗАМОК", "ОКЕАН", "ДЖУНГЛИ"]

HIERARCHY_TREE = {
    "КОСМОС":    {"level": 0, "parent": None,      "so3": np.array([0.0,  0.0,  0.0])},
    "ОКЕАН":     {"level": 0, "parent": None,      "so3": np.array([0.8, -1.0,  0.5])},
    "ГОРА":      {"level": 1, "parent": "КОСМОС",  "so3": np.array([-1.0, 0.5,  0.0])},
    "ПЛАНЕТА":   {"level": 1, "parent": "КОСМОС",  "so3": np.array([0.5, -0.2,  0.1])},
    "ДЖУНГЛИ":   {"level": 1, "parent": "ОКЕАН",   "so3": np.array([-1.2, -0.8, 0.2])},
    "КИБЕРПАНК": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([1.2,  0.8, -0.3])},
    "НЕБОСКРЕБ": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([1.5,  0.0,  1.0])},
    "ЗАМОК":     {"level": 2, "parent": "ГОРА",    "so3": np.array([-0.5, 0.3,  0.4])},
}

np.random.seed(42)
Q_RAW, _ = np.linalg.qr(np.random.randn(NUM_CHANNELS, 4))
SUBSPACE_BASIS = {
    0: Q_RAW[:, 0],  # Macro
    1: Q_RAW[:, 1],  # Meso
    2: Q_RAW[:, 2],  # Micro
    3: Q_RAW[:, 3]
}

CONCEPT_SPATIAL_OFFSETS = {
    name: np.linspace(-1.4 + i * 0.4, 1.4 - i * 0.25, NUM_CHANNELS, dtype=np.float32)
    for i, name in enumerate(ALL_CONCEPT_NAMES)
}

class AutonomousAgentProcess(mp.Process):
    def __init__(self, shm, num_concepts=8):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.num_concepts = num_concepts
        self.names = ALL_CONCEPT_NAMES[:num_concepts]

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}')
            outlets.append(StreamOutlet(info))

        print(f"🤖 [500Hz AGENT] Started Autonomous Dynamic Multi-Combination Engine...")
        start_time = time.time()
        mode_timer = time.time()
        active_demo_state = 0 # 0=Single, 1=Flat Pair, 2=Nest A>B, 3=Inverted B>A, 4=Deep 3-Level

        regional_delays = [0.0, 0.035, 0.070, 0.105]

        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            is_calib = self.shm['is_calibrating'].value

            phi_theta = TWO_PI * 6.0 * t_vec
            phi_delta = TWO_PI * 1.5 * t_vec
            theta_phase_norm = (phi_theta % TWO_PI) / TWO_PI

            low_gamma_sig  = np.zeros((NUM_CHANNELS, CHUNK_SIZE))
            high_gamma_sig = np.zeros((NUM_CHANNELS, CHUNK_SIZE))
            so3_active     = np.array([0.0, 0.0, 0.0])
            u_rank         = SUBSPACE_BASIS[0]

            # ==================================================================
            # 1. РЕЖИМ КАЛИБРОВКИ: Чистая запись без рипплов строго по запросу
            # ==================================================================
            if is_calib:
                tgt_idx = int(self.shm['calib_target_idx'].value)
                cur_name = self.names[tgt_idx]
                cur_level = HIERARCHY_TREE[cur_name]["level"]
                so3_active = HIERARCHY_TREE[cur_name]["so3"]
                u_rank = SUBSPACE_BASIS[cur_level]
                offset_ch = CONCEPT_SPATIAL_OFFSETS[cur_name]

                spatial_phase_ch = (COORDS_X * so3_active[0] + COORDS_Y * so3_active[1] + COORDS_Z * so3_active[2]) * 0.18
                spatial_phase_ch += u_rank * 0.40 + offset_ch

                f_gamma = 35.0 + tgt_idx * 5.0
                w_late = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)
                gamma_wave = np.sin(TWO_PI * f_gamma * t_vec[None, :] + spatial_phase_ch[:, None])
                low_gamma_sig = gamma_wave * w_late[None, :] * 6.0
                high_gamma_sig.fill(0.0)

                state_desc = f"CALIBRATION LTM: [{cur_name}]"
                satisfaction = 1.0
                boredom, frustration = 0.0, 0.0
                is_recursive_state = False

            # ==================================================================
            # 2. АКТИВНЫЙ РЕЖИМ: Автономный цикл по всем видам комбинаций
            # ==================================================================
            else:
                # Переключаем демонстрационную комбинацию каждые 10 секунд
                if time.time() - mode_timer > 10.0:
                    active_demo_state = (active_demo_state + 1) % 5
                    mode_timer = time.time()

                # --- Вариант 0: ОДИНОЧНЫЙ КОНЦЕПТ [ГОРА] (K=1.0) ---
                if active_demo_state == 0:
                    cur_name = "ГОРА"
                    tgt_idx = self.names.index(cur_name)
                    state_desc = "STATE 0: SINGLE CONCEPT [ГОРА] (K=1)"
                    so3_active = HIERARCHY_TREE[cur_name]["so3"]
                    u_rank = SUBSPACE_BASIS[1]
                    offset_ch = CONCEPT_SPATIAL_OFFSETS[cur_name]
                    spatial_phase_ch = (COORDS_X * so3_active[0] + COORDS_Y * so3_active[1] + COORDS_Z * so3_active[2]) * 0.18 + u_rank * 0.4 + offset_ch
                    w_late = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)
                    low_gamma_sig = np.sin(TWO_PI * 55.0 * t_vec[None, :] + spatial_phase_ch[:, None]) * w_late[None, :] * 5.5
                    high_gamma_sig.fill(0.0)
                    is_recursive_state = False

                # --- Вариант 1: ПЛОСКИЙ СПИСОК [ГОРА + ОКЕАН] (K=1.0, Сиблинги) ---
                elif active_demo_state == 1:
                    state_desc = "STATE 1: FLAT CO-OCCURRENCE [ГОРА + ОКЕАН] (K=1)"
                    # Чередование концептов по Дельте без риппл-вложения
                    in_delta_half = np.sin(phi_delta) >= 0.0
                    tgt_idx = self.names.index("ГОРА") if in_delta_half[-1] else self.names.index("ОКЕАН")
                    cur_name = self.names[tgt_idx]
                    u_rank = SUBSPACE_BASIS[1] # Один базис!
                    w_late = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)
                    for step_t in range(CHUNK_SIZE):
                        act_n = "ГОРА" if in_delta_half[step_t] else "ОКЕАН"
                        so3_c = HIERARCHY_TREE[act_n]["so3"]
                        off_c = CONCEPT_SPATIAL_OFFSETS[act_n]
                        sp_c = (COORDS_X * so3_c[0] + COORDS_Y * so3_c[1] + COORDS_Z * so3_c[2]) * 0.18 + u_rank * 0.4 + off_c
                        f_g = 55.0 if act_n == "ГОРА" else 65.0
                        low_gamma_sig[:, step_t] = np.sin(TWO_PI * f_g * t_vec[step_t] + sp_c) * w_late[step_t] * 5.5
                    high_gamma_sig.fill(0.0)
                    is_recursive_state = False

                # --- Вариант 2: ВЛОЖЕННОСТЬ А ⊃ Б [ЗАМОК НА ГОРЕ] (K=2.5, Causal Lead > 0) ---
                elif active_demo_state == 2:
                    state_desc = "STATE 2: RECURSION [ЗАМОК НА ГОРЕ] (Гора ⊃ Замок)"
                    tgt_idx = self.names.index("ЗАМОК")
                    dp_so3 = HIERARCHY_TREE["ЗАМОК"]["so3"] - HIERARCHY_TREE["ГОРА"]["so3"]
                    w_early = np.exp(-((theta_phase_norm - 0.25)**2) / 0.015)
                    w_late  = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)

                    # Риппл 150 Гц на ранней фазе (Гора = родитель)
                    phase_ripple = (COORDS_X * dp_so3[0] + COORDS_Y * dp_so3[1] + COORDS_Z * dp_so3[2]) * 0.25
                    high_gamma_sig = np.sin(TWO_PI * 150.0 * t_vec[None, :] + phase_ripple[:, None]) * w_early[None, :] * 6.0

                    # Гамма 60 Гц на поздней фазе (Замок = потомок в Micro)
                    sp_castle = (COORDS_X * HIERARCHY_TREE["ЗАМОК"]["so3"][0] + COORDS_Y * HIERARCHY_TREE["ЗАМОК"]["so3"][1]) * 0.18 + SUBSPACE_BASIS[2] * 0.45 + CONCEPT_SPATIAL_OFFSETS["ЗАМОК"]
                    low_gamma_sig = np.sin(TWO_PI * 60.0 * t_vec[None, :] + sp_castle[:, None]) * w_late[None, :] * 5.5
                    so3_active = HIERARCHY_TREE["ЗАМОК"]["so3"]
                    u_rank = SUBSPACE_BASIS[2]
                    is_recursive_state = True

                # --- Вариант 3: ИНВЕРТИРОВАННАЯ ВЛОЖЕННОСТЬ Б ⊃ А [ГОРА В ЗАМКЕ] (K=2.5, Causal Lead < 0) ---
                elif active_demo_state == 3:
                    state_desc = "STATE 3: INVERTED RECURSION [ГОРА В ЗАМКЕ] (Замок ⊃ Гора)"
                    tgt_idx = self.names.index("ГОРА")
                    # Инверсия вектора позы и фазового градиента
                    dp_so3 = HIERARCHY_TREE["ГОРА"]["so3"] - HIERARCHY_TREE["ЗАМОК"]["so3"]
                    w_early = np.exp(-((theta_phase_norm - 0.25)**2) / 0.015)
                    w_late  = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)

                    # Инвертированный риппл (Замок = родитель)
                    phase_ripple = (COORDS_X * dp_so3[0] + COORDS_Y * dp_so3[1] + COORDS_Z * dp_so3[2]) * 0.25
                    high_gamma_sig = np.sin(TWO_PI * 150.0 * t_vec[None, :] + phase_ripple[:, None]) * w_early[None, :] * 6.0

                    # Гора = потомок
                    sp_mount = (COORDS_X * HIERARCHY_TREE["ГОРА"]["so3"][0] + COORDS_Y * HIERARCHY_TREE["ГОРА"]["so3"][1]) * 0.18 + SUBSPACE_BASIS[2] * 0.45 + CONCEPT_SPATIAL_OFFSETS["ГОРА"]
                    low_gamma_sig = np.sin(TWO_PI * 55.0 * t_vec[None, :] + sp_mount[:, None]) * w_late[None, :] * 5.5
                    so3_active = HIERARCHY_TREE["ГОРА"]["so3"]
                    u_rank = SUBSPACE_BASIS[2]
                    is_recursive_state = True

                # --- Вариант 4: ГЛУБОКАЯ 3-УРОВНЕВАЯ РЕКУРСИЯ [КОСМОС ⊃ ПЛАНЕТА ⊃ КИБЕРПАНК] (K=3.8) ---
                else:
                    state_desc = "STATE 4: DEEP 3-LEVEL RECURSION [КОСМОС ⊃ ПЛАНЕТА ⊃ КИБЕРПАНК] (K=4)"
                    tgt_idx = self.names.index("КИБЕРПАНК")
                    w_macro = np.exp(-((theta_phase_norm - 0.15)**2) / 0.01)
                    w_meso  = np.exp(-((theta_phase_norm - 0.45)**2) / 0.01)
                    w_micro = np.exp(-((theta_phase_norm - 0.80)**2) / 0.01)

                    # Рипплы сшивают 3 уровня одновременно
                    dp1 = HIERARCHY_TREE["ПЛАНЕТА"]["so3"] - HIERARCHY_TREE["КОСМОС"]["so3"]
                    dp2 = HIERARCHY_TREE["КИБЕРПАНК"]["so3"] - HIERARCHY_TREE["ПЛАНЕТА"]["so3"]
                    pr1 = (COORDS_X * dp1[0] + COORDS_Y * dp1[1]) * 0.20
                    pr2 = (COORDS_X * dp2[0] + COORDS_Y * dp2[1]) * 0.25

                    high_gamma_sig = (
                        np.sin(TWO_PI * 130.0 * t_vec[None, :] + pr1[:, None]) * w_macro[None, :] * 4.0 +
                        np.sin(TWO_PI * 170.0 * t_vec[None, :] + pr2[:, None]) * w_meso[None, :] * 5.0
                    )
                    sp_cyber = (COORDS_X * HIERARCHY_TREE["КИБЕРПАНК"]["so3"][0]) * 0.15 + SUBSPACE_BASIS[2] * 0.45 + CONCEPT_SPATIAL_OFFSETS["КИБЕРПАНК"]
                    low_gamma_sig = np.sin(TWO_PI * 48.0 * t_vec[None, :] + sp_cyber[:, None]) * w_micro[None, :] * 5.5
                    so3_active = HIERARCHY_TREE["КИБЕРПАНК"]["so3"]
                    u_rank = SUBSPACE_BASIS[2]
                    is_recursive_state = True

                satisfaction = 0.95
                boredom = (time.time() - mode_timer) / 10.0
                frustration = 0.0

            # Передача телеметрии
            self.shm['agent_mode'].value = state_desc.split(":")[0].encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['agent_desc'].value = state_desc.encode('utf-8')[:255].ljust(256, b'\x00')
            self.shm['agent_mood'].value = b"RECURSIVE" if is_recursive_state else b"FLAT"
            self.shm['target_idx'].value = int(tgt_idx)
            self.shm['satisfaction'].value = float(satisfaction)
            self.shm['boredom'].value = float(boredom)
            self.shm['frustration'].value = float(frustration)
            self.shm['is_recursive'].value = bool(is_recursive_state)

            # Эмиссия физических пакетов на 4 электродных узла
            for node_i in range(NUM_DEVICES):
                delay = regional_delays[node_i]
                noise = np.random.normal(0, 0.035, (NUM_CHANNELS, CHUNK_SIZE))

                raw_sig = (
                    3.0 * np.sin(phi_theta + delay)[None, :] +
                    2.0 * np.sin(phi_delta + delay)[None, :] +
                    low_gamma_sig +
                    high_gamma_sig +
                    np.sin(TWO_PI * 22.0 * t_vec)[None, :] * 1.8 +
                    noise
                )

                if node_i == 2 and is_recursive_state:  # AFz передает направляющий риппл
                    raw_sig += high_gamma_sig * 1.4

                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)

class SyntheticAutonomousAgent:
    def __init__(self, num_concepts=8):
        ctx = mp.get_context('spawn')
        self.num_concepts = num_concepts
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [1.0 / num_concepts] * num_concepts + [0.0] * (256 - num_concepts)),
            'target_idx': ctx.Value('i', 0),
            'agent_mode': ctx.Array('c', 128),
            'agent_desc': ctx.Array('c', 256),
            'agent_mood': ctx.Array('c', 128),
            'satisfaction': ctx.Value('d', 1.0),
            'boredom': ctx.Value('d', 0.0),
            'frustration': ctx.Value('d', 0.0),
            'is_recursive': ctx.Value('b', False)
        }
        self.process = AutonomousAgentProcess(self.shm, num_concepts=num_concepts)
        self.process.start()

    def update_visual_state(self, probs):
        for i in range(min(self.num_concepts, len(probs))):
            self.shm['clip_probs'][i] = float(probs[i])

    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)

    def get_telemetry(self):
        return (
            self.shm['agent_mode'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['agent_desc'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['agent_mood'].value.decode('utf-8').replace('\x00', '').strip(),
            int(self.shm['target_idx'].value),
            float(self.shm['satisfaction'].value),
            float(self.shm['boredom'].value),
            float(self.shm['frustration'].value),
            bool(self.shm['is_recursive'].value)
        )

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
