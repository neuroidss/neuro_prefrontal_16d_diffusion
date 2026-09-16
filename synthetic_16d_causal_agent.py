#!/usr/bin/env python3
"""
🤖 SYNTHETIC SWARM ENGINE v1800.0 (PURE STIGMERGY & ZERO METAGAME)
- ПОЛНОСТЬЮ ИСКЛЮЧЕН МЕТАГЕЙМ: боты не знают о существовании других ботов.
- Никаких проверок is_solo или чтения чужих target_idx.
- Взаимодействие происходит ИСКЛЮЧИТЕЛЬНО через общий холст (Active Inference).
"""

import time
import math
import traceback
import threading
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 500.0
CHUNK_SIZE = 20
NUM_CHANNELS = 16
NUM_DEVICES = 4
TWO_PI = 2.0 * math.pi
VIDEO_BUFFER_LEN = 16

COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)

COORDS_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)

COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0)).astype(np.float32)

ALL_NAMES = ["КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ", "ГОРА", "ЗАМОК", "ОКЕАН", "ДЖУНГЛИ"]

HIERARCHY_TREE = {
    "КОСМОС":    {"level": 0, "parent": None,      "so3": np.array([ 0.0,  0.0,  0.0], dtype=np.float32)},
    "ОКЕАН":     {"level": 0, "parent": None,      "so3": np.array([ 0.8, -1.0,  0.5], dtype=np.float32)},
    "ГОРА":      {"level": 1, "parent": "КОСМОС",  "so3": np.array([-1.0,  0.5,  0.0], dtype=np.float32)},
    "ПЛАНЕТА":   {"level": 1, "parent": "КОСМОС",  "so3": np.array([ 0.5, -0.2,  0.1], dtype=np.float32)},
    "ДЖУНГЛИ":   {"level": 1, "parent": "ОКЕАН",   "so3": np.array([-1.2, -0.8,  0.2], dtype=np.float32)},
    "КИБЕРПАНК": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([ 1.2,  0.8, -0.3], dtype=np.float32)},
    "НЕБОСКРЕБ": {"level": 2, "parent": "ПЛАНЕТА", "so3": np.array([ 1.5,  0.0,  1.0], dtype=np.float32)},
    "ЗАМОК":     {"level": 2, "parent": "ГОРА",    "so3": np.array([-0.5,  0.3,  0.4], dtype=np.float32)},
}

ASSIGNABLE_CONCEPT_INDICES = [1, 4, 2, 5, 3, 6, 7, 0]

_eye_basis = np.eye(NUM_CHANNELS, dtype=np.float32)
SUBSPACE_BASIS = {i: _eye_basis[:, i] for i in range(4)}

CONCEPT_SPATIAL_OFFSETS = {
    name: np.linspace(-1.4 + i * 0.4, 1.4 - i * 0.25, NUM_CHANNELS, dtype=np.float32)
    for i, name in enumerate(ALL_NAMES)
}

class FullDynamicHardcodedBot:
    def __init__(self, bot_id: int, name_idx: int, num_concepts: int = 8):
        self.bot_id = bot_id
        self.target_idx = name_idx
        self.target_name = ALL_NAMES[name_idx]
        self.num_concepts = num_concepts
        self.frustration = 0.0
        self.state = "DICTATOR"

    def step(self, world_probs: np.ndarray):
        """
        ЧИСТАЯ СТИГМЕРГИЯ: Единственный источник информации — world_probs с холста.
        Никаких знаний о других ботах, их числе или их целях.
        """
        target_conf = float(world_probs[self.target_idx]) if self.target_idx < len(world_probs) else 0.0
        dominant_idx = int(np.argmax(world_probs))
        dominant_conf = float(world_probs[dominant_idx])

        # 1. Аттрактор захвачен: на холсте отображается целевой концепт
        if target_conf >= 0.35:
            self.frustration = max(0.0, self.frustration - 0.10)
            self.state = "LOCKED"
            return self.state, None, self.frustration

        # 2. На холсте доминирует стабильный чужой мир (среда сопротивляется воле агента)
        if dominant_idx != self.target_idx and dominant_conf >= 0.30:
            # Ошибка предсказания накапливает напряжение/фрустрацию
            self.frustration += 0.035
            if self.frustration > 1.0:
                # Фазовый переход в симбиоз: встраиваемся в победившую реальность
                self.state = "SYMBIOSIS"
                self.frustration = 0.5
                so3_world = HIERARCHY_TREE[ALL_NAMES[dominant_idx]]["so3"]
                so3_goal = HIERARCHY_TREE[self.target_name]["so3"]
                dp_so3 = so3_goal - so3_world
                return self.state, dp_so3, self.frustration
            else:
                self.state = "DICTATOR"
                return self.state, None, self.frustration

        # 3. На холсте шум / пустота / неясное состояние
        # Симбиоз с шумом невозможен, продолжаем утверждать свою волю
        self.frustration = max(0.0, self.frustration - 0.02)
        self.state = "DICTATOR"
        return self.state, None, self.frustration

    def generate_waves(self, state: str, dp_so3: np.ndarray | None, t_vec: np.ndarray, theta_norm: np.ndarray):
        freq = 35.0 + self.target_idx * 5.0
        so3_active = HIERARCHY_TREE[self.target_name]["so3"]
        offset_ch = CONCEPT_SPATIAL_OFFSETS[self.target_name]

        spatial_phase_ch = (COORDS_X * so3_active[0] + COORDS_Y * so3_active[1] + COORDS_Z * so3_active[2]) * 0.18 + SUBSPACE_BASIS[1] * 0.40 + offset_ch
        w_late = np.exp(-((theta_norm - 0.75)**2) / 0.02)
        
        # В LOCKED мягко удерживаем аттрактор, в DICTATOR давим на максимуме
        amp = 3.0 if state == "LOCKED" else 6.0
        low_gamma = np.sin(TWO_PI * freq * t_vec[None, :] + spatial_phase_ch[:, None]) * w_late[None, :] * amp

        high_ripple = np.zeros_like(low_gamma)
        if state == "SYMBIOSIS" and dp_so3 is not None:
            # При симбиозе излучаем рипплы на ранней фазе теты (0.25) для перестройки референтного фрейма
            w_early = np.exp(-((theta_norm - 0.25)**2) / 0.015)
            phase_ripple = (COORDS_X * dp_so3[0] + COORDS_Y * dp_so3[1] + COORDS_Z * dp_so3[2]) * 0.25
            high_ripple = np.sin(TWO_PI * 150.0 * t_vec[None, :] + phase_ripple[:, None]) * w_early[None, :] * 4.0

        return low_gamma, high_ripple


class FullJepaVideoAgent:
    def __init__(self, bot_id: int, name_idx: int, jepa_wrapper, num_concepts: int = 8):
        self.bot_id = bot_id
        self.target_idx = name_idx
        self.target_name = ALL_NAMES[name_idx]
        self.jepa = jepa_wrapper
        self.num_concepts = num_concepts
        self.dim = getattr(jepa_wrapper, "jepa_dim", 2048) if jepa_wrapper else 2048
        self.video_buffer = []
        self.state = "DICTATOR"
        self.frustration = 0.0

    def push_frame(self, frame_rgb: np.ndarray):
        self.video_buffer.append(frame_rgb)
        if len(self.video_buffer) > VIDEO_BUFFER_LEN:
            self.video_buffer.pop(0)

    def evaluate_and_plan(self, world_probs: np.ndarray):
        """
        ЧИСТАЯ СТИГМЕРГИЯ: Оценка текущего состояния визуального мира без метагейма.
        """
        if len(self.video_buffer) < 2:
            return "DICTATOR", None, 0.0

        try:
            target_conf = float(world_probs[self.target_idx]) if self.target_idx < len(world_probs) else 0.0
            dominant_idx = int(np.argmax(world_probs))
            dominant_conf = float(world_probs[dominant_idx])

            if target_conf >= 0.35:
                self.frustration = max(0.0, self.frustration - 0.10)
                self.state = "LOCKED"
                return "LOCKED", None, self.frustration

            if dominant_idx != self.target_idx and dominant_conf >= 0.30:
                self.frustration += 0.035
                if self.frustration > 1.0:
                    self.state = "SYMBIOSIS"
                    self.frustration = 0.5
                    so3_world = HIERARCHY_TREE[ALL_NAMES[dominant_idx]]["so3"]
                    so3_goal = HIERARCHY_TREE[self.target_name]["so3"]
                    dp_so3 = so3_goal - so3_world
                    return "SYMBIOSIS", dp_so3, self.frustration
                else:
                    self.state = "DICTATOR"
                    return "DICTATOR", None, self.frustration

            self.frustration = max(0.0, self.frustration - 0.02)
            self.state = "DICTATOR"
            return "DICTATOR", None, self.frustration

        except Exception:
            return "DICTATOR", None, 0.0

    def generate_waves(self, state: str, dp: np.ndarray | None, t_vec: np.ndarray, theta_norm: np.ndarray):
        freq = 35.0 + self.target_idx * 5.0
        so3_active = HIERARCHY_TREE[self.target_name]["so3"]
        offset_ch = CONCEPT_SPATIAL_OFFSETS[self.target_name]

        spatial_phase_ch = (COORDS_X * so3_active[0] + COORDS_Y * so3_active[1] + COORDS_Z * so3_active[2]) * 0.18 + SUBSPACE_BASIS[1] * 0.40 + offset_ch
        w_late = np.exp(-((theta_norm - 0.75)**2) / 0.02)
        amp = 3.0 if state == "LOCKED" else 6.0
        low_gamma = np.sin(TWO_PI * freq * t_vec[None, :] + spatial_phase_ch[:, None]) * w_late[None, :] * amp

        high_ripple = np.zeros_like(low_gamma)
        if state == "SYMBIOSIS" and dp is not None:
            w_early = np.exp(-((theta_norm - 0.25)**2) / 0.015)
            phase_ripple = (COORDS_X * dp[0] + COORDS_Y * dp[1] + COORDS_Z * dp[2]) * 0.25
            high_ripple = np.sin(TWO_PI * 150.0 * t_vec[None, :] + phase_ripple[:, None]) * w_early[None, :] * 4.0

        return low_gamma, high_ripple


class AutonomousSwarmProcess(mp.Process):
    def __init__(self, shm, num_hardcoded: int, num_jepa: int, num_concepts: int = 8):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.num_hardcoded = num_hardcoded
        self.num_jepa = num_jepa
        self.num_concepts = num_concepts

    def run(self):
        try:
            outlets = [
                StreamOutlet(StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}'))
                for i in range(NUM_DEVICES)
            ]

            jepa_wrapper = None
            if self.num_jepa > 0:
                try:
                    from vla_jepa_wrapper import VLA_JEPA_Wrapper
                    jepa_wrapper = VLA_JEPA_Wrapper(port=6001)
                except Exception:
                    pass

            self.bots = []
            assigned_ptr = 0

            for i in range(self.num_hardcoded):
                concept_idx = ASSIGNABLE_CONCEPT_INDICES[assigned_ptr % len(ASSIGNABLE_CONCEPT_INDICES)]
                self.bots.append(FullDynamicHardcodedBot(
                    bot_id=assigned_ptr + 1, name_idx=concept_idx, num_concepts=self.num_concepts
                ))
                assigned_ptr += 1

            for i in range(self.num_jepa):
                concept_idx = ASSIGNABLE_CONCEPT_INDICES[assigned_ptr % len(ASSIGNABLE_CONCEPT_INDICES)]
                self.bots.append(FullJepaVideoAgent(
                    bot_id=assigned_ptr + 1, name_idx=concept_idx, jepa_wrapper=jepa_wrapper, num_concepts=self.num_concepts
                ))
                assigned_ptr += 1

            start_time = time.time()
            regional_delays = [0.0, 0.035, 0.070, 0.105]
            bot_evals = {bot.bot_id: ("DICTATOR", None, 0.0) for bot in self.bots}
            eval_lock = threading.Lock()

            # Асинхронное планирование JEPA
            def async_jepa_worker():
                while self.shm['is_running'].value:
                    try:
                        if not self.shm['is_calibrating'].value:
                            raw_buf = np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8)
                            if np.any(raw_buf > 0):
                                frame_copy = raw_buf.reshape(384, 512, 3).copy()
                                cur_wprobs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)
                                for bot in self.bots:
                                    if isinstance(bot, FullJepaVideoAgent):
                                        bot.push_frame(frame_copy)
                                        res = bot.evaluate_and_plan(cur_wprobs)
                                        with eval_lock:
                                            bot_evals[bot.bot_id] = res
                    except Exception:
                        pass
                    time.sleep(0.1)

            if self.num_jepa > 0:
                t_jepa = threading.Thread(target=async_jepa_worker, daemon=True)
                t_jepa.start()

            self.shm['is_swarm_ready'].value = True
            last_hardcoded_plan = 0.0

            # Основной цикл генерации LFP сигналов (500 Гц)
            while self.shm['is_running'].value:
                dt = CHUNK_SIZE / FS
                t_now = time.time() - start_time
                t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False, dtype=np.float32)
                theta_phase_norm = ((TWO_PI * 6.0 * t_vec) % TWO_PI) / TWO_PI

                is_calib = self.shm['is_calibrating'].value
                calib_idx = int(self.shm['calib_target_idx'].value)
                world_probs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)

                low_gamma_sig = np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32)
                high_gamma_sig = np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32)

                owners = [-1] * 8

                if is_calib:
                    cur_name = ALL_NAMES[calib_idx]
                    so3_active = HIERARCHY_TREE[cur_name]["so3"]
                    offset_ch = CONCEPT_SPATIAL_OFFSETS[cur_name]

                    spatial_phase_ch = (COORDS_X * so3_active[0] + COORDS_Y * so3_active[1] + COORDS_Z * so3_active[2]) * 0.18 + SUBSPACE_BASIS[1] * 0.40 + offset_ch
                    w_late = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)
                    low_gamma_sig = np.sin(TWO_PI * (35.0 + calib_idx * 5.0) * t_vec[None, :] + spatial_phase_ch[:, None]) * w_late[None, :] * 6.0
                    status_str = f"CALIBRATING: [{cur_name}]"
                else:
                    if t_now - last_hardcoded_plan >= (1.0 / 6.0):
                        last_hardcoded_plan = t_now
                        with eval_lock:
                            for bot in self.bots:
                                if isinstance(bot, FullDynamicHardcodedBot):
                                    bot_evals[bot.bot_id] = bot.step(world_probs)

                    status_str = f"SWARM(H:{self.num_hardcoded},J:{self.num_jepa}):"
                    with eval_lock:
                        current_evals = dict(bot_evals)

                    for bot in self.bots:
                        owners[bot.target_idx] = bot.bot_id
                        state, dp, frust = current_evals.get(bot.bot_id, ("DICTATOR", None, 0.0))
                        
                        lg, hg = bot.generate_waves(state, dp, t_vec, theta_phase_norm)
                        low_gamma_sig += lg
                        high_gamma_sig += hg

                        prefix = "H" if isinstance(bot, FullDynamicHardcodedBot) else "J"
                        status_str += f" [{prefix}{bot.bot_id}-{bot.target_name[:4]}:{state[:3]}|F:{frust:.2f}]"

                    low_gamma_sig = np.clip(low_gamma_sig, -6.0, 6.0)
                    high_gamma_sig = np.clip(high_gamma_sig, -6.0, 6.0)

                for i in range(8):
                    self.shm['concept_owners'][i] = owners[i]

                self.shm['agent_desc'].value = status_str.encode('utf-8', errors='replace')[:250]

                has_active_signal = is_calib or (len(self.bots) > 0)

                for node_i in range(NUM_DEVICES):
                    if not has_active_signal:
                        raw_sig = np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32)
                    else:
                        delay = regional_delays[node_i]
                        noise = np.random.normal(0, 0.035, (NUM_CHANNELS, CHUNK_SIZE))
                        raw_sig = (
                            3.0 * np.sin(TWO_PI * 6.0 * t_vec + delay)[None, :] +
                            2.0 * np.sin(TWO_PI * 1.5 * t_vec + delay)[None, :] +
                            low_gamma_sig +
                            high_gamma_sig +
                            np.sin(TWO_PI * 22.0 * t_vec)[None, :] * 1.8 +
                            noise
                        )
                        if node_i == 2 and not is_calib and np.any(high_gamma_sig):
                            raw_sig += high_gamma_sig * 1.5

                    outlets[node_i].push_chunk(raw_sig.T.tolist())

                time.sleep(dt)

        except Exception as err:
            print(f"❌ [SWARM CRASH]: {err}")
            traceback.print_exc()


class SyntheticAutonomousAgent:
    def __init__(self, num_concepts: int = 8, num_hardcoded: int = 1, num_jepa: int = 0):
        ctx = mp.get_context('spawn')
        self.num_concepts = num_concepts
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'is_swarm_ready': ctx.Value('b', False),
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [0.0] * 256),
            'raw_rgb_frame': ctx.Array('B', 384 * 512 * 3),
            'agent_desc': ctx.Array('c', 256),
            'concept_owners': ctx.Array('i', [-1] * 8)
        }
        self.process = AutonomousSwarmProcess(
            self.shm, num_hardcoded=num_hardcoded, num_jepa=num_jepa, num_concepts=num_concepts
        )
        self.process.start()

    def is_ready(self) -> bool:
        return bool(self.shm['is_swarm_ready'].value)

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def update_visual_state(self, visual_data):
        if isinstance(visual_data, np.ndarray):
            if visual_data.ndim == 3:
                np.copyto(np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8), visual_data.reshape(-1))
            elif visual_data.ndim == 1:
                for i in range(min(self.num_concepts, len(visual_data))):
                    self.shm['clip_probs'][i] = float(visual_data[i])

    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)

    def get_telemetry(self):
        return self.shm['agent_desc'].value.decode('utf-8', errors='replace').replace('\x00', '').strip()

    def get_owners(self):
        return list(self.shm['concept_owners'][:8])

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
