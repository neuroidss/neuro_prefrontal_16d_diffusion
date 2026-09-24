#!/usr/bin/env python3
"""
🤖 SYNTHETIC ACTIVE INFERENCE SWARM v2500.0 (FULL 8-REGION NEUROANATOMICAL SUITE)
- ПОЛНЫЙ АНСАМБЛЬ: F3 (Порядок), F4 (Хаос), AFz (Тор), Fpz (Ветвление), FCz (Действия).
- Никакой регион не удаляется! Все 5 префронтальных центров работают одновременно.
- НИКАКОГО МЕТАГЕЙМИНГА: Агенты видят мир через CLIP/JEPA и выдают чистое 16-канальное ЭЭГ.
- АКТИВНЫЙ ВЫВОД: Агенты исследуют аффордансы (НАД ⊃, ВНУТРЬ ➔, МЕЖДУ ∥, ПОД ⊂).
"""

import time
import math
import json
import os
import traceback
import threading
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250.0
CHUNK_SIZE = 20
NUM_CHANNELS = 16
NUM_DEVICES = 8  # Расширен до 8 каналов (полный мозг)
TWO_PI = 2.0 * math.pi
VIDEO_BUFFER_LEN = 16

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
COORDS_Z = np.sqrt(np.maximum(100.0 - COORDS_X**2 - COORDS_Y**2, 0.0)).astype(np.float32)

ALL_NAMES = ["ГОРА",  "ДЖУНГЛИ", "ЗАМОК", "ОКЕАН", "КОСМОС", "ПЛАНЕТА", "КИБЕРПАНК", "НЕБОСКРЕБ"]

CONCEPT_SCALE_PRIORS = {
    "КОСМОС": 3, "ПЛАНЕТА": 3,
    "ОКЕАН": 2, "ГОРА": 2, "ДЖУНГЛИ": 2,
    "ЗАМОК": 1, "НЕБОСКРЕБ": 1,
    "КИБЕРПАНК": 0
}

def torus_geodesic_distance(u1, v1, u2, v2):
    du = abs(u1 - u2) % TWO_PI
    if du > math.pi: du = TWO_PI - du
    dv = abs(v1 - v2) % TWO_PI
    if dv > math.pi: dv = TWO_PI - dv
    return math.sqrt(du**2 + dv**2)

class CorticalMontage:
    def __init__(self, config_dict):
        self.devices = config_dict.get('montage', {}).get('devices', [])

    def get_nodes_by_names(self, names: list) -> list[int]:
        matched_ids = []
        for dev in self.devices:
            if any(dev['name'].startswith(n) for n in names):
                matched_ids.append(dev['id'])
        return list(set(matched_ids))

    def get_nodes_by_radius(self, anchor_xyz: list, radius: float) -> list[int]:
        matched_ids = []
        anchor = np.array(anchor_xyz)
        for dev in self.devices:
            dist = np.linalg.norm(np.array(dev['coords']) - anchor)
            if dist <= radius:
                matched_ids.append(dev['id'])
        return list(set(matched_ids))
        
    def check_feature(self, active_nodes: list, feature_name: str) -> bool:
        for dev in self.devices:
            if dev['id'] in active_nodes and feature_name in dev['name']:
                return True
        return False

class BaseActiveAgent:
    def __init__(self, bot_id: int, initial_idx: int, active_nodes: list, montage: CorticalMontage, num_concepts: int = 8):
        self.bot_id = bot_id
        self.current_idx = initial_idx
        self.original_idx = initial_idx
        self.target_name = ALL_NAMES[initial_idx % len(ALL_NAMES)]
        self.num_concepts = num_concepts
        self.active_nodes = active_nodes
        
        # Полный спектр когнитивных центров:
        self.has_f3  = montage.check_feature(active_nodes, "F3")  # Левая PFC (Порядок)
        self.has_f4  = montage.check_feature(active_nodes, "F4")  # Правая PFC (Хаос)
        self.has_afz = montage.check_feature(active_nodes, "AFz") # Тор Джанаты (Смыслы)
        self.has_fpz = montage.check_feature(active_nodes, "Fpz") # Ветвление BA 10
        self.has_fcz = montage.check_feature(active_nodes, "FCz") # Действия SMA
        
        all_other = [i for i in range(num_concepts) if i != initial_idx]
        np.random.seed(bot_id * 101)
        self.plan_b_queue = list(np.random.permutation(all_other))
        
        self.role = "LEADER"
        self.partner_idx = None
        self.sprt_log_evidence = 0.0
        self.u_torus = (initial_idx * TWO_PI / num_concepts)
        self.v_torus = ((initial_idx * 3) * TWO_PI / num_concepts) % TWO_PI
        self.strategy_attempts = 0
        self.strategies_pool = ["SUPER_PARENT", "EMBED_INTO", "PEER", "SUB_CHILD"]

    def evaluate_affordance_strategy(self, dominant_idx: int):
        my_name = ALL_NAMES[self.original_idx % len(ALL_NAMES)]
        dom_name = ALL_NAMES[dominant_idx % len(ALL_NAMES)]
        my_scale = CONCEPT_SCALE_PRIORS.get(my_name, 1)
        dom_scale = CONCEPT_SCALE_PRIORS.get(dom_name, 1)
        
        if my_scale > dom_scale: return "SUPER_PARENT"
        elif my_scale < dom_scale: return "EMBED_INTO"
        else: return "PEER" if (self.bot_id % 2 == 0) else "SUB_CHILD"

    def test_next_hypothesis(self, dominant_idx: int):
        if dominant_idx == self.original_idx:
            self.role = "LEADER"
            self.partner_idx = None
            return

        self.strategy_attempts += 1
        chosen = self.evaluate_affordance_strategy(dominant_idx) if self.strategy_attempts == 1 else self.strategies_pool[(self.strategy_attempts - 1) % len(self.strategies_pool)]
        self.role = chosen
        self.partner_idx = dominant_idx
        
        dom_u = (dominant_idx * TWO_PI / self.num_concepts)
        dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
        my_name = ALL_NAMES[self.original_idx % len(ALL_NAMES)]
        dom_name = ALL_NAMES[dominant_idx % len(ALL_NAMES)]

        if self.role == "SUPER_PARENT":
            self.u_torus = (dom_u - math.pi / 4.0) % TWO_PI
            self.v_torus = (dom_v - math.pi / 4.0) % TWO_PI
            rel_sym = "⊃"
        elif self.role == "EMBED_INTO":
            self.u_torus = (dom_u + math.pi / 3.0) % TWO_PI
            self.v_torus = (dom_v + math.pi / 3.0) % TWO_PI
            rel_sym = "➔"
        elif self.role == "PEER":
            self.u_torus = (dom_u + math.pi / 2.0) % TWO_PI
            self.v_torus = (dom_v) % TWO_PI
            rel_sym = "∥"
        else:
            self.u_torus = (dom_u + math.pi) % TWO_PI
            self.v_torus = (dom_v + math.pi) % TWO_PI
            rel_sym = "⊂"

        self.target_name = f"{my_name} {rel_sym} {dom_name}"

    def trigger_fpz_cognitive_branch(self):
        if not self.has_fpz: return False 
        self.sprt_log_evidence = 0.0
        self.strategy_attempts = 0
        old_idx = self.current_idx
        self.plan_b_queue.append(old_idx)
        
        next_idx = self.plan_b_queue.pop(0)
        if self.partner_idx is not None and next_idx == self.partner_idx and len(self.plan_b_queue) > 0:
            self.plan_b_queue.append(next_idx)
            next_idx = self.plan_b_queue.pop(0)

        self.current_idx = next_idx
        self.original_idx = next_idx
        self.target_name = ALL_NAMES[self.current_idx % len(ALL_NAMES)]
        self.partner_idx = None 
        self.u_torus = (self.current_idx * TWO_PI / self.num_concepts)
        self.v_torus = ((self.current_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
        self.role = "LEADER"
        print(f"🔀 [Fpz ВЕТВЛЕНИЕ]: Агент {self.bot_id} сменил фокус: {ALL_NAMES[old_idx % len(ALL_NAMES)]} ➔ {self.target_name}!")
        return True

    def generate_waves(self, role: str, dp_so3: np.ndarray | None, t_vec: np.ndarray, theta_norm: np.ndarray):
        freq = 35.0 + (self.current_idx % 8) * 3.5
        offset_ch = np.linspace(-1.0, 1.0, NUM_CHANNELS, dtype=np.float32)
        spatial_phase_ch = (COORDS_X * math.cos(self.u_torus) + COORDS_Y * math.sin(self.v_torus)) * 0.18 + offset_ch

        if role in ["LEADER", "SUPER_PARENT"]:
            target_phase, amp = 0.22, 5.2
        elif role == "EMBED_INTO":
            target_phase, amp = 0.45, 4.8
        elif role == "PEER":
            target_phase, amp = 0.50, 4.4
        else:
            target_phase, amp = 0.78, 4.0

        w_theta = np.exp(-((theta_norm - target_phase)**2) / 0.025)
        low_gamma = np.sin(TWO_PI * freq * t_vec[None, :] + spatial_phase_ch[:, None]) * w_theta[None, :] * amp

        high_ripple = np.zeros_like(low_gamma)
        if dp_so3 is not None:
            w_ripple = np.exp(-((theta_norm - ((target_phase + 0.12) % 1.0))**2) / 0.015)
            phase_ripple = (COORDS_X * dp_so3[0] + COORDS_Y * dp_so3[1] + COORDS_Z * dp_so3[2]) * 0.25
            high_ripple = np.sin(TWO_PI * 89.5 * t_vec[None, :] + phase_ripple[:, None]) * w_ripple[None, :] * 5.0
            
        return low_gamma, high_ripple

class FullDynamicHardcodedBot(BaseActiveAgent):
    def step(self, effective_probs: np.ndarray):
        target_p = float(effective_probs[self.current_idx % len(effective_probs)]) if len(effective_probs) > 0 else 0.0
        dominant_idx = int(np.argmax(effective_probs)) if len(effective_probs) > 0 else 0
        dominant_p = float(effective_probs[dominant_idx]) if len(effective_probs) > 0 else 0.0

        if target_p >= 0.25:
            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.20)
            self.role = "LEADER"
            self.partner_idx = None
            self.strategy_attempts = 0
            return self.role, None, self.sprt_log_evidence

        if dominant_idx != self.current_idx and dominant_p >= 0.25:
            dom_u = (dominant_idx * TWO_PI / self.num_concepts)
            dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
            d_torus = torus_geodesic_distance(self.u_torus, self.v_torus, dom_u, dom_v) if self.has_afz else math.pi

            evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4)) * (0.5 + d_torus / math.pi)
            self.sprt_log_evidence += 0.008 * evidence_step

            if int(self.sprt_log_evidence * 10) % 8 == 0:
                self.test_next_hypothesis(dominant_idx)

            if self.sprt_log_evidence > 1.45:
                if self.trigger_fpz_cognitive_branch():
                    return "BRANCH_HOP", None, self.sprt_log_evidence
                else:
                    self.sprt_log_evidence = 1.45

            du = self.u_torus - dom_u
            dv = self.v_torus - dom_v
            dp_so3 = np.array([math.sin(du), math.cos(dv), math.sin(du + dv)], dtype=np.float32)
            if self.role in ["SUB_CHILD", "EMBED_INTO"]: dp_so3 = -dp_so3
            return self.role, dp_so3, self.sprt_log_evidence

        self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
        return self.role, None, self.sprt_log_evidence

class FullJepaVideoAgent(BaseActiveAgent):
    def __init__(self, bot_id: int, initial_idx: int, active_nodes: list, montage: CorticalMontage, jepa_wrapper, num_concepts: int = 8):
        super().__init__(bot_id, initial_idx, active_nodes, montage, num_concepts)
        self.jepa = jepa_wrapper
        self.video_buffer = []
        self.last_jepa_energy = 1.0

    def push_frame(self, frame_rgb: np.ndarray):
        self.video_buffer.append(frame_rgb)
        if len(self.video_buffer) > VIDEO_BUFFER_LEN: self.video_buffer.pop(0)

    def step(self, effective_probs: np.ndarray):
        if len(self.video_buffer) < 2: return self.role, None, self.sprt_log_evidence
        try:
            target_p = float(effective_probs[self.current_idx % len(effective_probs)]) if len(effective_probs) > 0 else 0.0
            dominant_idx = int(np.argmax(effective_probs)) if len(effective_probs) > 0 else 0
            dominant_p = float(effective_probs[dominant_idx]) if len(effective_probs) > 0 else 0.0

            if self.jepa is not None and len(self.video_buffer) >= 2:
                z_cur = self.jepa.encode_world_state(self.video_buffer[-1])
                self.last_jepa_energy = float(z_cur.abs().mean().item()) if hasattr(z_cur, 'abs') else 0.5

            if target_p >= 0.25:
                self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.25)
                self.role = "LEADER"
                self.partner_idx = None
                self.strategy_attempts = 0
                return self.role, None, self.sprt_log_evidence

            if dominant_idx != self.current_idx and dominant_p >= 0.25:
                dom_u = (dominant_idx * TWO_PI / self.num_concepts)
                dom_v = ((dominant_idx * 3) * TWO_PI / self.num_concepts) % TWO_PI
                d_torus = torus_geodesic_distance(self.u_torus, self.v_torus, dom_u, dom_v) if self.has_afz else math.pi

                evidence_step = math.log((dominant_p + 1e-4) / (target_p + 1e-4)) * (0.6 + 0.4 * self.last_jepa_energy)
                self.sprt_log_evidence += 0.009 * evidence_step

                if int(self.sprt_log_evidence * 10) % 7 == 0:
                    self.test_next_hypothesis(dominant_idx)

                if self.sprt_log_evidence > 1.45:
                    if self.trigger_fpz_cognitive_branch():
                        return "BRANCH_HOP", None, self.sprt_log_evidence
                    else:
                        self.sprt_log_evidence = 1.45

                du = self.u_torus - dom_u
                dv = self.v_torus - dom_v
                dp_so3 = np.array([math.sin(du), math.cos(dv), math.sin(du + dv)], dtype=np.float32)
                if self.role in ["SUB_CHILD", "EMBED_INTO"]: dp_so3 = -dp_so3
                return self.role, dp_so3, self.sprt_log_evidence

            self.sprt_log_evidence = max(0.0, self.sprt_log_evidence - 0.05)
            return self.role, None, self.sprt_log_evidence
        except Exception:
            return self.role, None, 0.0

class AutonomousSwarmProcess(mp.Process):
    def __init__(self, shm, config_path: str, num_hardcoded: int, num_jepa: int, num_concepts: int = 8, sps: float = 250.0):
        super().__init__()
        self.daemon = True
        self.shm = shm
        self.config_path = config_path
        self.num_hardcoded = num_hardcoded
        self.num_jepa = num_jepa
        self.num_concepts = num_concepts
        self.sps = float(sps)

    def run(self):
        FS = self.sps
        try:
            config_dict = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            else:
                # ВСЕ 8 РЕГИОНОВ МОЗГА В МОНТАЖЕ:
                config_dict = {
                    "montage": {
                        "system": "10-20-extended",
                        "devices": [
                            {"id": 0, "name": "F3",  "coords": [-4.0, 3.0, 5.0]},  # Левая PFC (Порядок)
                            {"id": 1, "name": "F4",  "coords": [4.0, 3.0, 5.0]},   # Правая PFC (Хаос)
                            {"id": 2, "name": "AFz", "coords": [0.0, 6.0, 4.0]},   # Тор Джанаты
                            {"id": 3, "name": "Fpz", "coords": [0.0, 8.0, 2.0]},   # Ветвление BA10
                            {"id": 4, "name": "FCz", "coords": [0.0, 3.0, 7.0]},   # Действия SMA
                            {"id": 5, "name": "Pz",  "coords": [0.0, -6.0, 6.0]},  # Теменная
                            {"id": 6, "name": "Oz",  "coords": [0.0, -10.0, 1.0]}, # Зрительная
                            {"id": 7, "name": "Cz",  "coords": [0.0, 0.0, 9.0]}    # Сенсомоторная
                        ]
                    },
                    "agents": []
                }
                all_targets = ["F3", "F4", "AFz", "Fpz", "FCz"]
                if self.num_hardcoded > 0:
                    config_dict["agents"].append({
                        "type": "hardcoded", "count": self.num_hardcoded,
                        "profile": "CLI_Hardcoded_Bot", "bind_mode": "names",
                        "bind_target": all_targets
                    })
                if self.num_jepa > 0:
                    config_dict["agents"].append({
                        "type": "jepa", "count": self.num_jepa,
                        "profile": "CLI_JEPA_Bot", "bind_mode": "names",
                        "bind_target": all_targets
                    })

            montage = CorticalMontage(config_dict)

            # Открываем потоки под все доступные узлы
            num_outlets = min(NUM_DEVICES, len(config_dict.get('montage', {}).get('devices', [])))
            outlets = [
                StreamOutlet(StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_node_{i}'))
                for i in range(num_outlets)
            ]

            jepa_wrapper = None
            if any(a.get('type') == 'jepa' for a in config_dict.get('agents', [])):
                try:
                    from vla_jepa_wrapper import VLA_JEPA_Wrapper
                    jepa_wrapper = VLA_JEPA_Wrapper(port=6001)
                except Exception as e:
                    print(f"⚠️ [SWARM] VLA-JEPA недоступна: {e}")

            rng = np.random.RandomState(42)
            available_slots = list(rng.permutation(self.num_concepts))
            self.bots = []
            bot_idx = 1

            for agent_cfg in config_dict.get('agents', []):
                count = agent_cfg.get('count', 1)
                for _ in range(count):
                    c_idx = available_slots[(bot_idx - 1) % len(available_slots)]
                    mode = agent_cfg.get('bind_mode', 'names')
                    if mode == 'radius':
                        tgt = agent_cfg.get('bind_target', {})
                        active_nodes = montage.get_nodes_by_radius(tgt.get('anchor', [0,0,0]), tgt.get('radius', 5.0))
                    else:
                        active_nodes = montage.get_nodes_by_names(agent_cfg.get('bind_target', []))
                    
                    if not active_nodes: active_nodes = [0, 1, 2, 3, 4]
                    
                    if agent_cfg.get('type') == 'jepa':
                        bot = FullJepaVideoAgent(bot_idx, c_idx, active_nodes, montage, jepa_wrapper, self.num_concepts)
                    else:
                        bot = FullDynamicHardcodedBot(bot_idx, c_idx, active_nodes, montage, self.num_concepts)
                        
                    self.bots.append(bot)
                    bot_idx += 1

            start_time = time.time()
            regional_delays = [0.0, 0.035, 0.070, 0.105, 0.140, 0.175, 0.210, 0.245]

            def async_jepa_worker():
                while self.shm['is_running'].value:
                    try:
                        if not self.shm['is_calibrating'].value:
                            raw_buf = np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8)
                            if np.any(raw_buf > 0):
                                frame_copy = raw_buf.reshape(384, 512, 3).copy()
                                for bot in self.bots:
                                    if isinstance(bot, FullJepaVideoAgent):
                                        bot.push_frame(frame_copy)
                    except Exception:
                        pass
                    time.sleep(0.08)

            if any(isinstance(b, FullJepaVideoAgent) for b in self.bots):
                threading.Thread(target=async_jepa_worker, daemon=True).start()

            self.shm['is_swarm_ready'].value = True

            while self.shm['is_running'].value:
                dt = CHUNK_SIZE / FS
                t_now = time.time() - start_time
                t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False, dtype=np.float32)
                theta_phase_norm = ((TWO_PI * 6.0 * t_vec) % TWO_PI) / TWO_PI

                is_calib = self.shm['is_calibrating'].value
                world_probs = np.array(self.shm['clip_probs'][:self.num_concepts], dtype=np.float32)
                
                if self.shm['sensory_sub'].value:
                    priors = np.array(self.shm['swarm_priors'][:self.num_concepts], dtype=np.float32)
                    effective_probs = world_probs * 0.5 + priors * 0.5
                else:
                    effective_probs = world_probs

                device_lg = [np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32) for _ in range(num_outlets)]
                device_hg = [np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32) for _ in range(num_outlets)]
                owners = [-1] * 8
                
                calib_target = ALL_NAMES[self.shm['calib_target_idx'].value % len(ALL_NAMES)]
                status_str = "SWARM:" if not is_calib else f"CALIB: [{calib_target}]"

                if not is_calib:
                    for b_idx, bot in enumerate(self.bots):
                        role, dp, sprt_val = bot.step(effective_probs)
                        owners[bot.current_idx % 8] = bot.bot_id
                        
                        if role == "SUPER_PARENT" and bot.partner_idx is not None:
                            self.shm['parent_map'][bot.partner_idx] = bot.original_idx
                        elif role in ["EMBED_INTO", "SUB_CHILD"] and bot.partner_idx is not None:
                            self.shm['parent_map'][bot.original_idx] = bot.partner_idx
                        else:
                            self.shm['parent_map'][bot.original_idx] = -1

                        lg, hg = bot.generate_waves(role, dp, t_vec, theta_phase_norm)
                        for node_i in bot.active_nodes:
                            if node_i < num_outlets:
                                device_lg[node_i] += lg
                                device_hg[node_i] += hg

                        prefix = "H" if isinstance(bot, FullDynamicHardcodedBot) else "J"
                        nodes_str = "".join([str(n) for n in bot.active_nodes])
                        role_short = role[:4]
                        status_str += f" [{prefix}{bot.bot_id}({nodes_str}):{role_short}|{bot.target_name[:7]}]"

                for i in range(num_outlets):
                    device_lg[i] = np.clip(device_lg[i], -6.0, 6.0)
                    device_hg[i] = np.clip(device_hg[i], -6.0, 6.0)

                for i in range(8): self.shm['concept_owners'][i] = owners[i]
                self.shm['agent_desc'].value = status_str.encode('utf-8', errors='replace')[:250]

                for node_i in range(num_outlets):
                    if is_calib and not self.bots:
                        raw_sig = np.zeros((NUM_CHANNELS, CHUNK_SIZE), dtype=np.float32)
                    else:
                        delay = regional_delays[node_i % len(regional_delays)]
                        noise = np.random.normal(0, 0.035, (NUM_CHANNELS, CHUNK_SIZE))
                        raw_sig = (
                            3.0 * np.sin(TWO_PI * 6.0 * t_vec + delay)[None, :] +
                            2.0 * np.sin(TWO_PI * 1.5 * t_vec + delay)[None, :] +
                            device_lg[node_i] +
                            device_hg[node_i] +
                            np.sin(TWO_PI * 22.0 * t_vec)[None, :] * 1.8 +
                            noise
                        )
                        if node_i == 2 and not is_calib and np.any(device_hg[node_i]):
                            raw_sig += device_hg[node_i] * 1.5

                    outlets[node_i].push_chunk(raw_sig.T.tolist())

                time.sleep(dt)

        except Exception as err:
            print(f"❌ [SWARM CRASH]: {err}")
            traceback.print_exc()

class SyntheticAutonomousAgent:
    def __init__(self, config_path: str = "swarm_config.json", num_hardcoded: int = 4, num_jepa: int = 4, num_concepts: int = 8, sps: float = 250.0):
        ctx = mp.get_context('spawn')
        self.num_concepts = num_concepts
        self.shm = {
            'is_running': ctx.Value('b', True),
            'is_calibrating': ctx.Value('b', True),
            'is_swarm_ready': ctx.Value('b', False),
            'sensory_sub': ctx.Value('b', False), 
            'calib_target_idx': ctx.Value('i', 0),
            'clip_probs': ctx.Array('d', [0.0] * 256),
            'swarm_priors': ctx.Array('d', [0.0] * 256), 
            'raw_rgb_frame': ctx.Array('B', 384 * 512 * 3),
            'agent_desc': ctx.Array('c', 256),
            'concept_owners': ctx.Array('i', [-1] * 8),
            'parent_map': ctx.Array('i', [-1] * 16) 
        }
        self.process = AutonomousSwarmProcess(
            self.shm, config_path, num_hardcoded=num_hardcoded, num_jepa=num_jepa, num_concepts=num_concepts, sps=sps
        )
        self.process.start()

    def is_ready(self) -> bool: return bool(self.shm['is_swarm_ready'].value)
    def is_alive(self) -> bool: return self.process.is_alive()
    def set_sensory_substitution(self, state: bool): self.shm['sensory_sub'].value = bool(state)
    def update_swarm_priors(self, priors: np.ndarray):
        for i in range(min(self.num_concepts, len(priors))): self.shm['swarm_priors'][i] = float(priors[i])
    def update_visual_state(self, visual_data):
        if isinstance(visual_data, np.ndarray):
            if visual_data.ndim == 3:
                np.copyto(np.frombuffer(self.shm['raw_rgb_frame'].get_obj(), dtype=np.uint8), visual_data.reshape(-1))
            elif visual_data.ndim == 1:
                for i in range(min(self.num_concepts, len(visual_data))): self.shm['clip_probs'][i] = float(visual_data[i])
    def get_parent_map(self):
        pmap = list(self.shm['parent_map'][:self.num_concepts])
        return {i: (p if p != -1 else None) for i, p in enumerate(pmap)}
    def set_calibration_target(self, active: bool, tgt_idx: int = 0):
        self.shm['is_calibrating'].value = bool(active)
        self.shm['calib_target_idx'].value = int(tgt_idx)
    def get_telemetry(self): return self.shm['agent_desc'].value.decode('utf-8', errors='replace').replace('\x00', '').strip()
    def get_owners(self): return list(self.shm['concept_owners'][:8])
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
