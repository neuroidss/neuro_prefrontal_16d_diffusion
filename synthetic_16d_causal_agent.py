#!/usr/bin/env python3
"""
🤖 SYNTHETIC CAUSAL AGENT v700.0 (AUTONOMOUS STIGMERGIC SWARM)
- Никакой магии движка. Агенты полностью изолированы.
- Агенты смотрят ТОЛЬКО на картинку Диффузии (через CLIP_probs).
- Если их концепт не на картинке, они встраивают его в то, что видят (Симбиоз).
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 500.0
CHUNK_SIZE = 20
NUM_CHANNELS = 16
NUM_DEVICES = 4
TWO_PI = 2.0 * math.pi

COORDS_X = np.array([10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
                     -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
                      2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71], dtype=np.float32)
COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0))

ALL_NAMES = ["КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ", "ГОРА", "ЗАМОК", "ОКЕАН", "ДЖУНГЛИ"]

HIERARCHY_TREE = {
    "КОСМОС":    {"so3": np.array([0.0,  0.0,  0.0])},
    "ОКЕАН":     {"so3": np.array([0.8, -1.0,  0.5])},
    "ГОРА":      {"so3": np.array([-1.0, 0.5,  0.0])},
    "ПЛАНЕТА":   {"so3": np.array([0.5, -0.2,  0.1])},
    "ДЖУНГЛИ":   {"so3": np.array([-1.2, -0.8, 0.2])},
    "КИБЕРПАНК": {"so3": np.array([1.2,  0.8, -0.3])},
    "НЕБОСКРЕБ": {"so3": np.array([1.5,  0.0,  1.0])},
    "ЗАМОК":     {"so3": np.array([-0.5, 0.3,  0.4])},
}

np.random.seed(42)
Q_RAW, _ = np.linalg.qr(np.random.randn(NUM_CHANNELS, 4))
SUBSPACE_BASIS = {0: Q_RAW[:, 0], 1: Q_RAW[:, 1], 2: Q_RAW[:, 2], 3: Q_RAW[:, 3]}

CONCEPT_SPATIAL_OFFSETS = {
    name: np.linspace(-1.4 + i * 0.4, 1.4 - i * 0.25, NUM_CHANNELS, dtype=np.float32)
    for i, name in enumerate(ALL_NAMES)
}

class BotAgent:
    def __init__(self, bot_id, name_idx, num_concepts):
        self.bot_id = bot_id
        self.target_idx = name_idx
        self.target_name = ALL_NAMES[name_idx]
        self.num_concepts = num_concepts
        self.frustration = 0.0
        self.state = "DICTATOR"
        self.parent_target_idx = None

    def step(self, world_probs):
        dominant_world_idx = int(np.argmax(world_probs))
        
        # Агент оценивает картинку диффузии
        if world_probs[self.target_idx] > 0.2:
            self.frustration = max(0.0, self.frustration - 0.05)
            self.state = "DICTATOR"
            self.parent_target_idx = None
        else:
            self.frustration += 0.05
            if self.frustration > 1.0:
                self.state = "SYMBIOSIS"
                self.parent_target_idx = dominant_world_idx
                self.frustration = 0.5 

        dp_so3 = None
        if self.state == "SYMBIOSIS" and self.parent_target_idx is not None:
            so3_world = HIERARCHY_TREE[ALL_NAMES[self.parent_target_idx]]["so3"]
            so3_goal = HIERARCHY_TREE[self.target_name]["so3"]
            dp_so3 = so3_goal - so3_world

        return self.state, dp_so3

class AutonomousSwarmProcess(mp.Process):
    def __init__(self, shm, num_concepts=8):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.num_concepts = num_concepts
        self.bots = [
            BotAgent(1, ALL_NAMES.index("ПЛАНЕТА"), num_concepts),
            BotAgent(2, ALL_NAMES.index("ГОРА"), num_concepts),
            BotAgent(3, ALL_NAMES.index("ЗАМОК"), num_concepts)
        ]

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}')
            outlets.append(StreamOutlet(info))

        print(f"🤖 [STIGMERGIC SWARM] Started generating LSL streams...")
        start_time = time.time()
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

            if is_calib:
                tgt_idx = int(self.shm['calib_target_idx'].value)
                cur_name = ALL_NAMES[tgt_idx]
                so3_active = HIERARCHY_TREE[cur_name]["so3"]
                offset_ch = CONCEPT_SPATIAL_OFFSETS[cur_name]

                spatial_phase_ch = (COORDS_X * so3_active[0] + COORDS_Y * so3_active[1] + COORDS_Z * so3_active[2]) * 0.18 + SUBSPACE_BASIS[1] * 0.40 + offset_ch
                w_late = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)
                low_gamma_sig = np.sin(TWO_PI * (35.0 + tgt_idx * 5.0) * t_vec[None, :] + spatial_phase_ch[:, None]) * w_late[None, :] * 6.0
                
                status_str = f"CALIBRATING: Generating pure signal for [{cur_name}]"
                
                owners = [-1] * 8
                owners[tgt_idx] = 0 
                for i in range(8): self.shm['concept_owners'][i] = owners[i]
            else:
                world_probs = np.array(self.shm['clip_probs'][:self.num_concepts])
                
                status_str = "SWARM:"
                owners = [-1] * 8
                
                human_tgt = int(self.shm['human_intent_idx'].value)
                owners[human_tgt] = 0
                
                for bot in self.bots:
                    state, dp_so3 = bot.step(world_probs)
                    owners[bot.target_idx] = bot.bot_id
                    
                    sp_goal = (COORDS_X * HIERARCHY_TREE[bot.target_name]["so3"][0] + COORDS_Y * HIERARCHY_TREE[bot.target_name]["so3"][1]) * 0.18 + SUBSPACE_BASIS[2] * 0.45 + CONCEPT_SPATIAL_OFFSETS[bot.target_name]
                    w_late = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)
                    low_gamma_sig += np.sin(TWO_PI * 60.0 * t_vec[None, :] + sp_goal[:, None]) * w_late[None, :] * 2.0

                    if state == "SYMBIOSIS" and dp_so3 is not None:
                        w_early = np.exp(-((theta_phase_norm - 0.25)**2) / 0.015)
                        phase_ripple = (COORDS_X * dp_so3[0] + COORDS_Y * dp_so3[1] + COORDS_Z * dp_so3[2]) * 0.25
                        high_gamma_sig += np.sin(TWO_PI * 150.0 * t_vec[None, :] + phase_ripple[:, None]) * w_early[None, :] * 3.0
                        status_str += f" [{bot.target_name} ⊃ SYM]"
                    else:
                        status_str += f" [{bot.target_name} -> DIC]"

                for i in range(8): self.shm['concept_owners'][i] = owners[i]

                low_gamma_sig = np.clip(low_gamma_sig, -6.0, 6.0)
                high_gamma_sig = np.clip(high_gamma_sig, -6.0, 6.0)

            self.shm['agent_desc'].value = status_str.encode('utf-8')[:255].ljust(256, b'\x00')

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
                if node_i == 2 and not is_calib and np.any(high_gamma_sig):
                    raw_sig += high_gamma_sig * 1.5
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
            'agent_desc': ctx.Array('c', 256),
            'concept_owners': ctx.Array('i', [-1] * 8),
            'human_intent_idx': ctx.Value('i', 0)
        }
        self.process = AutonomousSwarmProcess(self.shm, num_concepts=num_concepts)
        self.process.start()

    def update_visual_state(self, probs):
        for i in range(min(self.num_concepts, len(probs))):
            self.shm['clip_probs'][i] = float(probs[i])

    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)

    def set_human_intent(self, intent_idx: int):
        self.shm['human_intent_idx'].value = int(intent_idx)

    def get_telemetry(self):
        return self.shm['agent_desc'].value.decode('utf-8').replace('\x00', '').strip()

    def get_owners(self):
        return list(self.shm['concept_owners'][:8])

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
