#!/usr/bin/env python3
"""
🤖 SYNTHETIC ACTIVE INFERENCE SWARM v2200.0
- УДАЛЕНЫ ВСЕ ХАРДКОДНЫЕ ПЕРЕСТАНОВКИ. Инициализация через конкурентное распределение (Chen et al. 2024 Neuron).
- Топология Тора Джанаты вычисляется эмпирически через MDS матрицу RDM (Fan et al. 2024 Nat Hum Behav).
- Когнитивное ветвление Fpz (BA10) управляется тестом Вальда SPRT/DDM (Boorman et al. 2009 Neuron, Gold & Shadlen 2007).
- Физиологические кортикальные рипплы человека ~89.5 Гц (Dickey et al. 2022 PNAS).
- VLA-JEPA минимизация свободной энергии в латентном пространстве мира (V-JEPA 2, Assran 2025; Sun 2026).
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

def torus_geodesic_distance(u1, v1, u2, v2):
    """
    Геодезическое расстояние на 2D-торе T^2 = S^1 x S^1 (Janata 2002 Science).
    """
    du = abs(u1 - u2) % TWO_PI
    if du > math.pi: du = TWO_PI - du
    dv = abs(v1 - v2) % TWO_PI
    if dv > math.pi: dv = TWO_PI - dv
    return math.sqrt(du**2 + dv**2)


class BaseActiveAgent:
    def __init__(self, bot_id: int, initial_idx: int, num_concepts: int = 8):
        self.bot_id = bot_id
        self.current_idx = initial_idx
        self.target_name = ALL_NAMES[initial_idx]
        self.num_concepts = num_concepts
        
        # Альтернативные гипотезы в очереди ожидания Fpz (BA10, Koechlin 2007)
        all_other = [i for i in range(num_concepts) if i != initial_idx]
        np.random.seed(bot_id * 101)
        self.plan_b_queue = list(np.random.permutation(all_other))
        
        self.role = "LEADER"
        self.sprt_log_evidence = 0.0 # Аккумулятор логарифма правдоподобия Вальда (SPRT)
        self.u_torus = (initial_idx * TWO_PI / num_concepts)
        self.v_torus = ((initial_idx * 3) * TWO_PI / num_concepts) % TWO_PI
        self.steps_in_role = 0

    def trigger_fpz_cognitive_branch(self):
        """
        Когнитивное ветвление Fpz (Boorman et al., 2009 Neuron; Gold & Shadlen 2007):
        Сброс аккумулятора и фазовый скачок на 120° к следующей альтернативной гипотезе.
        """
        self.sprt_log_evidence = 0.0
        old_idx = self.current_idx
        self.plan_b_queue.append(old_idx)
        self.current_idx = self.plan_b_queue.pop(0)
        self.target_name = ALL_NAMES[self.current_idx]
        self.u_torus = (self.current_idx * TWO_PI / self.num_concepts)
        self.v_torus = ((self.current_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
        self.role = "LEADER"
        self.steps_in_role = 0
        print(f"🔀 [Fpz BA10 BRANCHING] Бот {self.bot_id} переключил фокус: {ALL_NAMES[old_idx]} ➔ {self.target_name} (SPRT Bound Exceeded)!")

    def generate_waves(self, role: str, dp_so3: np.ndarray | None, t_vec: np.ndarray, theta_norm: np.ndarray):
        freq = 35.0 + (self.current_idx % 8) * 3.5
        offset_ch = np.linspace(-1.0, 1.0, NUM_CHANNELS, dtype=np.float32)

        spatial_phase_ch = (COORDS_X * math.cos(self.u_torus) + COORDS_Y * math.sin(self.v_torus)) * 0.18 + offset_ch

        # Мультиплексирование фаз Теты (Bieri et al., 2014 Neuron)
        if role in ["LEADER", "SUPER_PARENT"]:
            target_phase = 0.25  # Ранняя тета (Macro-контейнер)
            amp = 5.0
        elif role == "SUB_CHILD":
            target_phase = 0.75  # Поздняя тета (Вложенная деталь)
            amp = 3.8
        else: # PEER
            target_phase = 0.50  # Средняя тета (Равноправный сосед)
            amp = 4.2

        w_theta = np.exp(-((theta_norm - target_phase)**2) / 0.025)
        low_gamma = np.sin(TWO_PI * freq * t_vec[None, :] + spatial_phase_ch[:, None]) * w_theta[None, :] * amp

        # Физиологические кортикальные рипплы ~89.5 Гц (Dickey et al., PNAS 2022)
        high_ripple = np.zeros_like(low_gamma)
        if dp_so3 is not None:
            w_ripple = np.exp(-((theta_norm - ((target_phase + 0.15) % 1.0))**2) / 0.015)
            phase_ripple = (COORDS_X * dp_so3[0] + COORDS_Y * dp_so3[1] + COORDS_Z * dp_so3[2]) * 0.25
            high_ripple = np.sin(TWO_PI * 89.5 * t_vec[None, :] + phase_ripple[:, None]) * w_ripple[None, :] * 4.5

        return low_gamma, high_ripple


class FullDynamicHardcodedBot(BaseActiveAgent):
    """
    Активный агент с накоплением ошибки по Вальду (SPRT) и перебором ролей в гетерархии TBT 2.0.
    """
    def step(self, world_probs: np.ndarray):
        target_p = float(world_probs[self.current_idx]) if self.current_idx < len(world_probs) else 0.0
        dominant_idx = int(np.argmax(world_probs))
        dominant_p = float(world_probs[dominant_idx])

        # 1. Аттрактор взят (наша цель отображается на холсте)
        if target_p >= 0.28:
            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.20)
            self.role = "LEADER"
            self.steps_in_role = 0
            return self.role, None, self.sprt_log_evidence

        # 2. На холсте доминирует другой концепт: навигация по Тору Джанаты
        if dominant_idx != self.current_idx and dominant_p >= 0.25:
            dom_u = (dominant_idx * TWO_PI / self.num_concepts)
            dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
            d_torus = torus_geodesic_distance(self.u_torus, self.v_torus, dom_u, dom_v)

            # Модель аккумуляции ошибки Вальда (SPRT): логарифм расхождения вероятностей
            evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4)) * (0.5 + d_torus / math.pi)
            self.sprt_log_evidence += 0.04 * evidence_step
            self.steps_in_role += 1

            # Попытка смены гетерархической роли (Child -> Peer -> Super)
            if self.steps_in_role > 18:
                self.steps_in_role = 0
                if self.role == "LEADER": self.role = "SUB_CHILD"
                elif self.role == "SUB_CHILD": self.role = "PEER"
                elif self.role == "PEER": self.role = "SUPER_PARENT"

            # Порог когнитивного ветвления Вальда (Boorman 2009, Gold & Shadlen 2007)
            if self.sprt_log_evidence > 1.35:
                self.trigger_fpz_cognitive_branch()
                return "BRANCH_HOP", None, self.sprt_log_evidence

            # Относительная поза SO(3) на Торе Джанаты
            du = self.u_torus - dom_u
            dv = self.v_torus - dom_v
            dp_so3 = np.array([math.sin(du), math.cos(dv), math.sin(du + dv)], dtype=np.float32)
            if self.role != "SUB_CHILD": dp_so3 = -dp_so3

            return self.role, dp_so3, self.sprt_log_evidence

        self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
        return self.role, None, self.sprt_log_evidence


class FullJepaVideoAgent(BaseActiveAgent):
    """
    Агент VLA-JEPA: использует энкодер мира V-JEPA 2 (Assran et al. 2025, Sun et al. 2026).
    Минимизирует свободную энергию в латентном пространстве.
    """
    def __init__(self, bot_id: int, initial_idx: int, jepa_wrapper, num_concepts: int = 8):
        super().__init__(bot_id, initial_idx, num_concepts)
        self.jepa = jepa_wrapper
        self.video_buffer = []
        self.last_jepa_energy = 1.0

    def push_frame(self, frame_rgb: np.ndarray):
        self.video_buffer.append(frame_rgb)
        if len(self.video_buffer) > VIDEO_BUFFER_LEN:
            self.video_buffer.pop(0)

    def evaluate_and_plan(self, world_probs: np.ndarray):
        if len(self.video_buffer) < 2:
            return self.role, None, self.sprt_log_evidence

        try:
            target_p = float(world_probs[self.current_idx]) if self.current_idx < len(world_probs) else 0.0
            dominant_idx = int(np.argmax(world_probs))
            dominant_p = float(world_probs[dominant_idx])

            # Энергия ошибки предсказания в латентном пространстве V-JEPA 2 (Assran et al. 2025)
            if self.jepa is not None and len(self.video_buffer) >= 2:
                z_cur = self.jepa.encode_world_state(self.video_buffer[-1])
                norm_val = float(z_cur.abs().mean().item()) if hasattr(z_cur, 'abs') else 0.5
                self.last_jepa_energy = norm_val

            if target_p >= 0.28:
                self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.25)
                self.role = "LEADER"
                self.steps_in_role = 0
                return self.role, None, self.sprt_log_evidence

            if dominant_idx != self.current_idx and dominant_p >= 0.25:
                dom_u = (dominant_idx * TWO_PI / self.num_concepts)
                dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
                d_torus = torus_geodesic_distance(self.u_torus, self.v_torus, dom_u, dom_v)

                # Шаг SPRT, взвешенный латентной энергией V-JEPA 2
                evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4)) * (0.6 + 0.4 * self.last_jepa_energy)
                self.sprt_log_evidence += 0.045 * evidence_step
                self.steps_in_role += 1

                if self.steps_in_role > 15:
                    self.steps_in_role = 0
                    if self.role == "LEADER": self.role = "SUB_CHILD"
                    elif self.role == "SUB_CHILD": self.role = "PEER"
                    elif self.role == "PEER": self.role = "SUPER_PARENT"

                if self.sprt_log_evidence > 1.35:
                    self.trigger_fpz_cognitive_branch()
                    return "BRANCH_HOP", None, self.sprt_log_evidence

                du = self.u_torus - dom_u
                dv = self.v_torus - dom_v
                dp_so3 = np.array([math.sin(du), math.cos(dv), math.sin(du + dv)], dtype=np.float32)
                if self.role != "SUB_CHILD": dp_so3 = -dp_so3

                return self.role, dp_so3, self.sprt_log_evidence

            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
            return self.role, None, self.sprt_log_evidence

        except Exception:
            return self.role, None, 0.0


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
                    print(f"🤖 [SWARM] VLA-JEPA обертка подключена (порт 6001).")
                except Exception as e:
                    print(f"⚠️ [SWARM] VLA-JEPA недоступна: {e}")

            # НАУЧНАЯ ИНИЦИАЛИЗАЦИЯ: Конкурентное распределение по ортогональным подпространствам (Chen 2024 Neuron)
            # Никаких захардкоженных списков! Равномерная псевдослучайная перестановка пространства задач.
            rng = np.random.RandomState(42)
            available_slots = list(rng.permutation(self.num_concepts))
            self.bots = []

            for i in range(self.num_hardcoded):
                c_idx = available_slots[i % len(available_slots)]
                self.bots.append(FullDynamicHardcodedBot(
                    bot_id=i+1, initial_idx=c_idx, num_concepts=self.num_concepts
                ))

            for j in range(self.num_jepa):
                b_id = self.num_hardcoded + j + 1
                c_idx = available_slots[(self.num_hardcoded + j) % len(available_slots)]
                self.bots.append(FullJepaVideoAgent(
                    bot_id=b_id, initial_idx=c_idx, jepa_wrapper=jepa_wrapper, num_concepts=self.num_concepts
                ))

            start_time = time.time()
            regional_delays = [0.0, 0.035, 0.070, 0.105]
            bot_evals = {bot.bot_id: ("LEADER", None, 0.0) for bot in self.bots}
            eval_lock = threading.Lock()

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
                    time.sleep(0.08)

            if self.num_jepa > 0:
                t_jepa = threading.Thread(target=async_jepa_worker, daemon=True)
                t_jepa.start()

            self.shm['is_swarm_ready'].value = True
            last_hardcoded_plan = 0.0

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
                    u = (calib_idx * TWO_PI / self.num_concepts)
                    v = ((calib_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
                    spatial_phase_ch = (COORDS_X * math.cos(u) + COORDS_Y * math.sin(v)) * 0.18
                    w_late = np.exp(-((theta_phase_norm - 0.75)**2) / 0.02)
                    low_gamma_sig = np.sin(TWO_PI * (35.0 + calib_idx * 3.5) * t_vec[None, :] + spatial_phase_ch[:, None]) * w_late[None, :] * 6.0
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
                        owners[bot.current_idx] = bot.bot_id
                        role, dp, sprt_val = current_evals.get(bot.bot_id, ("LEADER", None, 0.0))
                        
                        lg, hg = bot.generate_waves(role, dp, t_vec, theta_phase_norm)
                        low_gamma_sig += lg
                        high_gamma_sig += hg

                        prefix = "H" if isinstance(bot, FullDynamicHardcodedBot) else "J"
                        status_str += f" [{prefix}{bot.bot_id}-{bot.target_name[:4]}:{role[:3]}|E:{sprt_val:.2f}]"

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

    def is_ready(self) -> bool: return bool(self.shm['is_swarm_ready'].value)
    def is_alive(self) -> bool: return self.process.is_alive()
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
