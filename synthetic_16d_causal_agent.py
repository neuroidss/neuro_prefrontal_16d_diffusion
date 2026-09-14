#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: INVASIVE LAMINAR HIERARCHICAL ACTIVE INFERENCE AGENT (v400.0)
Strictly grounded in micro-electrode & intracranial primate/human electrophysiology.
+ ПЕРЕКЛЮЧАТЕЛЬ 500Hz: РЕКУРСИВНОЕ МУЛЬТИПЛЕКСИРОВАНИЕ ФАЗЫ vs ПЛОСКАЯ СУПЕРПОЗИЦИЯ
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 500.0        # 500 SPS
CHUNK_SIZE = 20   # 40 мс при 500 Гц
NUM_CHANNELS = 16
NUM_DEVICES = 4   # Node 0: F3, Node 1: F4, Node 2: AFz, Node 3: Fpz
TWO_PI = 2.0 * math.pi

COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)

COORDS_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)

COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0))

ALL_CONCEPT_NAMES = [
    "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ",
    "ГОРА", "ЗАМОК", "ОКЕАН", "ДЖУНГЛИ"
]

HIERARCHY_TREE = {
    "КОСМОС":    {"level": 0, "parent": None,      "so3": np.array([0.0,  0.0,  0.0])},
    "ОКЕАН":     {"level": 0, "parent": None,      "so3": np.array([0.8, -1.0,  0.5])},
    "ПЛАНЕТА":   {"level": 1, "parent": "КОСМОС",  "so3": np.array([0.5, -0.2,  0.1])},
    "ГОРА":      {"level": 1, "parent": "КОСМОС",  "so3": np.array([-1.0, 0.5,  0.0])},
    "ДЖУНГЛИ":   {"level": 1, "parent": "ОКЕАН",   "so3": np.array([-1.2, -0.8, 0.2])},
    "КИБЕРПАНК": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([1.2,  0.8, -0.3])},
    "НЕБОСКРЕБ": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([1.5,  0.0,  1.0])},
    "ЗАМОК":     {"level": 2, "parent": "ГОРА",    "so3": np.array([-0.5, 0.3,  0.4])},
}

np.random.seed(42)
Q_RAW, _ = np.linalg.qr(np.random.randn(NUM_CHANNELS, 4))
SUBSPACE_BASIS = {
    0: Q_RAW[:, 0],  
    1: Q_RAW[:, 1],  
    2: Q_RAW[:, 2],
    3: Q_RAW[:, 3]
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

        print(f"🤖 [500Hz AGENT] Started Invasive Laminar Engine ({self.num_concepts} concepts)...")

        start_time = time.time()
        tgt_idx = 0
        plan_b_idx = 1
        boredom = 0.0
        frustration = 0.0
        saccade_cooldown = 0.0
        theta_phase_offset = 0.0
        active_transition = "IDLE"

        regional_delays = [0.0, 0.035, 0.070, 0.105]

        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            is_calib = self.shm['is_calibrating'].value
            is_recursive = self.shm['is_recursive'].value
            clip_probs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)

            cur_name = self.names[tgt_idx]
            cur_node = HIERARCHY_TREE[cur_name]
            cur_level = cur_node["level"]

            children_indices = [i for i, n in enumerate(self.names) if HIERARCHY_TREE[n]["parent"] == cur_name]
            parent_name = cur_node["parent"]
            parent_idx = self.names.index(parent_name) if (parent_name and parent_name in self.names) else None
            sibling_indices = [i for i, n in enumerate(self.names) if HIERARCHY_TREE[n]["level"] == cur_level and i != tgt_idx]

            if is_calib:
                tgt_idx = int(self.shm['calib_target_idx'].value)
                mode = "CALIBRATION"
                state_desc = f"Calibration LTM Imprint: [{self.names[tgt_idx]}]"
                satisfaction, boredom, frustration = 1.0, 0.0, 0.0
                mood = "LEARNING"
                prediction_error = 0.0
            else:
                satisfaction = float(clip_probs[tgt_idx])
                prediction_error = 1.0 - satisfaction

                sorted_candidates = np.argsort(clip_probs)
                alt_idx = int(sorted_candidates[-2]) if sorted_candidates[-2] != tgt_idx else int(sorted_candidates[-3])
                plan_b_idx = alt_idx if alt_idx < self.num_concepts else (tgt_idx + 1) % self.num_concepts

                if saccade_cooldown > 0.0:
                    saccade_cooldown = max(0.0, saccade_cooldown - dt)
                    mode = active_transition
                    mood = "TRANSITION"
                    state_desc = f"{active_transition} -> [{self.names[tgt_idx]}] (L{cur_level})"
                elif satisfaction >= 0.70:
                    mode = f"HOLD ({int(satisfaction*100)}%)"
                    boredom += dt
                    frustration = max(0.0, frustration - dt * 2.5)
                    mood = "CONVERGED"
                    state_desc = f"Attractor Locked: [{self.names[tgt_idx]}] (L{cur_level})"
                else:
                    mode = "SEEKING"
                    frustration += dt * (prediction_error * 2.2)
                    boredom = max(0.0, boredom - dt * 2.0)
                    mood = "CONFLICT" if frustration > 4.0 else "SEARCH"
                    state_desc = f"Resolving [{self.names[tgt_idx]}] (PE: {prediction_error:.2f})"

                if saccade_cooldown <= 0.0:
                    if is_recursive:
                        if satisfaction >= 0.80 and boredom >= 2.5 and len(children_indices) > 0:
                            tgt_idx = children_indices[0]
                            saccade_cooldown, active_transition, boredom, frustration = 1.4, "ZOOM_IN", 0.0, 0.0
                            theta_phase_offset += math.pi * 0.5
                        elif frustration >= 5.0 and parent_idx is not None:
                            tgt_idx = parent_idx
                            saccade_cooldown, active_transition, boredom, frustration = 1.4, "ZOOM_OUT", 0.0, 0.0
                            theta_phase_offset += math.pi * 0.5
                        elif (boredom >= 4.0 or frustration >= 6.5):
                            tgt_idx = sibling_indices[0] if len(sibling_indices) > 0 and np.random.rand() > 0.3 else plan_b_idx
                            active_transition = "LATERAL_SACCADE"
                            saccade_cooldown, boredom, frustration = 1.0, 0.0, 0.0
                            theta_phase_offset += math.pi
                    else:
                        if boredom >= 2.0 or frustration >= 5.0:
                            tgt_idx = np.random.randint(0, self.num_concepts)
                            saccade_cooldown, active_transition, boredom, frustration = 1.0, "FLAT_JUMP", 0.0, 0.0
                            theta_phase_offset += math.pi

            self.shm['agent_mode'].value = mode.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['agent_desc'].value = state_desc.encode('utf-8')[:255].ljust(256, b'\x00')
            self.shm['agent_mood'].value = mood.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['target_idx'].value = int(tgt_idx)
            self.shm['satisfaction'].value = float(satisfaction)
            self.shm['boredom'].value = float(boredom / 4.0)
            self.shm['frustration'].value = float(frustration / 8.0)

            phi_theta = TWO_PI * 6.0 * t_vec + theta_phase_offset
            phi_delta = TWO_PI * 2.5 * t_vec
            hierarchical_scale = 0.3 + (cur_level * 0.45)

            if saccade_cooldown > 0.0:
                beta_amp, desync_noise = 0.4, 0.12
            else:
                beta_amp, desync_noise = 3.5, 0.01

            beta_wave = np.sin(TWO_PI * 22.0 * t_vec) * beta_amp
            alpha_wave = np.sin(TWO_PI * 10.0 * t_vec) * 1.5

            # Включение 150 Гц несущей для риппл-диапазона
            ripple_carrier = np.sin(TWO_PI * 150.0 * t_vec) * 4.0

            vec_current = HIERARCHY_TREE[cur_name]["so3"]
            vec_plan_b = HIERARCHY_TREE[self.names[plan_b_idx]]["so3"]

            # Мультиплексирование фазы в Тета-цикле
            theta_phase_norm = (phi_theta % TWO_PI) / TWO_PI
            subspace_idx = (np.floor(theta_phase_norm * 4.0).astype(int)) % 4

            for node_i in range(NUM_DEVICES):
                delay = regional_delays[node_i]
                
                if node_i == 0: so3_active, spatial_gain, extra_sig = vec_current, 1.2, 0.0
                elif node_i == 1: so3_active, spatial_gain, extra_sig = vec_current, 1.0, 0.0
                elif node_i == 2:
                    so3_active, spatial_gain = vec_current, 0.8
                    extra_sig = np.clip(prediction_error * 4.0, 0.0, 5.0) * np.sin(phi_theta + delay)
                else:
                    so3_active, spatial_gain, extra_sig = vec_plan_b, 0.7, 0.0

                base_spatial_phase = (
                    COORDS_X[:, None] * so3_active[0] +
                    COORDS_Y[:, None] * so3_active[1] +
                    COORDS_Z[:, None] * so3_active[2]
                ) * (0.15 * hierarchical_scale * spatial_gain)

                # ИСТИННАЯ РЕКУРСИЯ: фазовый вектор переключается каждые 90° Теты
                # ПЛОСКОСТЬ: фазовый вектор статичен
                if is_recursive:
                    subspace_mod = np.zeros((NUM_CHANNELS, CHUNK_SIZE))
                    for step_t in range(CHUNK_SIZE):
                        s_idx = subspace_idx[step_t]
                        subspace_mod[:, step_t] = SUBSPACE_BASIS[s_idx] * 0.50
                else:
                    subspace_mod = np.tile((SUBSPACE_BASIS[0] + SUBSPACE_BASIS[1]) * 0.25, (CHUNK_SIZE, 1)).T

                final_spatial_phase = base_spatial_phase + subspace_mod
                noise = np.random.normal(0, desync_noise, (NUM_CHANNELS, len(t_vec)))

                raw_sig = (
                    3.0 * np.sin(phi_theta + delay)[None, :] +
                    2.0 * np.sin(phi_delta + delay)[None, :] +
                    beta_wave[None, :] +
                    ripple_carrier[None, :] * np.sin(final_spatial_phase) +
                    alpha_wave[None, :] +
                    extra_sig +
                    noise
                )

                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)

class SyntheticAutonomousAgent:
    def __init__(self, num_concepts=8):
        ctx = mp.get_context('spawn')
        self.num_concepts = num_concepts
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'is_recursive': ctx.Value('b', True),
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [1.0 / num_concepts] * num_concepts + [0.0] * (256 - num_concepts)),
            'target_idx': ctx.Value('i', 0),
            'agent_mode': ctx.Array('c', 128),
            'agent_desc': ctx.Array('c', 256),
            'agent_mood': ctx.Array('c', 128),
            'satisfaction': ctx.Value('d', 0.0),
            'boredom': ctx.Value('d', 0.0),
            'frustration': ctx.Value('d', 0.0)
        }
        self.process = AutonomousAgentProcess(self.shm, num_concepts=num_concepts)
        self.process.start()

    def update_visual_state(self, probs):
        for i in range(min(self.num_concepts, len(probs))):
            self.shm['clip_probs'][i] = float(probs[i])

    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)
        
    def toggle_recursion(self):
        self.shm['is_recursive'].value = not self.shm['is_recursive'].value
        return self.shm['is_recursive'].value

    def get_telemetry(self):
        return (
            self.shm['agent_mode'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['agent_desc'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['agent_mood'].value.decode('utf-8').replace('\x00', '').strip(),
            self.shm['target_idx'].value,
            self.shm['satisfaction'].value,
            self.shm['boredom'].value,
            self.shm['frustration'].value,
            self.shm['is_recursive'].value
        )

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
