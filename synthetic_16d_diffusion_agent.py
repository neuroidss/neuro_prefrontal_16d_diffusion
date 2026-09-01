#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: CLOSED-LOOP 16D DIFFUSION AGENT (v111.0)
- Управляет 4-мя узлами LSL (AFz, F3, F4, Fpz) по 16 каналов (250 Hz).
- Плавное наведение на 4 полюса семантического базиса (Гора, Замок, Океан, Небоскреб).
- Детектор тупика (Stagnation) -> Ветвление Fpz -> Фазовый срыв (Phase Reset).
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250.0
CHUNK_SIZE = 10
NUM_CHANNELS = 16
NUM_DEVICES = 4
TWO_PI = 2.0 * math.pi

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
IS_CORE = np.hypot(COORDS_X, COORDS_Y) < 8.0

# 4 СЕМАНТИЧЕСКИХ ПОЛЮСА ДИФФУЗИИ
DIFFUSION_TARGETS = [
    {"name": "🏔️ ГОРА (MOUNTAIN)",       "g": 0.0, "s": 0.0, "desc": "F3=0.0 (Природа) | F4=0.0 (Свет)"},
    {"name": "🏰 ЗАМОК (CASTLE)",         "g": 1.0, "s": 0.0, "desc": "F3=1.0 (Камень)  | F4=0.0 (Свет)"},
    {"name": "🌊 ОКЕАН (OCEAN)",          "g": 0.0, "s": 1.0, "desc": "F3=0.0 (Вода)    | F4=1.0 (Шторм)"},
    {"name": "🏙️ НЕБОСКРЕБ (SKYSCRAPER)", "g": 1.0, "s": 1.0, "desc": "F3=1.0 (Стекло)   | F4=1.0 (Огни)"}
]

class AgentDiffusionProcess(mp.Process):
    def __init__(self, shm_dict):
        super().__init__()
        self.daemon = True
        self.shm = shm_dict

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_diff_node_{i}')
            outlets.append(StreamOutlet(info))

        print("🤖 [DIFFUSION AGENT] Запущен 16D агент семантического наведения (250 Hz)...")

        start_time = time.time()
        target_idx = 0
        state_mode = "EXPLORE"
        hold_timer = 0.0
        stagnation_timer = 0.0

        current_cmd = np.zeros((4, 4), dtype=np.float32)
        target_cmd = np.zeros((4, 4), dtype=np.float32)

        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            tgt = DIFFUSION_TARGETS[target_idx]
            honest_match = self.shm['honest_match'].value

            # --- КОГНИТИВНЫЙ АКТИВНЫЙ ВЫВОД ---
            if state_mode == "EXPLORE":
                # Наведение F3 (g) и F4 (s)
                target_cmd[0, 0] = (tgt["g"] * 2.0 - 1.0) # в диапазон [-1, 1]
                target_cmd[1, 0] = (tgt["s"] * 2.0 - 1.0)
                target_cmd[2, 3] = 0.3 # Нормальная жесткость AFz
                target_cmd[3, 2] = 0.2 # Тень слегка видна
                target_cmd[3, 3] = 0.0

                if honest_match >= 0.76:
                    state_mode = "HOLD"
                    hold_timer = 0.0
                    print(f"🎯 [AGENT] Захват аттрактора диффузии: {tgt['name']} (Match: {honest_match*100:.1f}%)")

            elif state_mode == "HOLD":
                hold_timer += dt
                # Накачиваем Сагитту Fpz (Тень становится плотной)
                target_cmd[3, 2] = float(np.clip(0.2 + (hold_timer / 3.5) * 0.8, 0.0, 1.0))
                target_cmd[3, 3] = 0.0

                if hold_timer >= 3.5:
                    state_mode = "SWITCH"
                    hold_timer = 0.0
                    print(f"💥 [PHASE RESET] Срыв фазы -> Переключение на следующую цель!")

            elif state_mode == "SWITCH":
                hold_timer += dt
                # Импульс срыва Fpz ry -> 1.0
                target_cmd[3, 3] = 1.0
                if hold_timer >= 0.35:
                    target_idx = (target_idx + 1) % len(DIFFUSION_TARGETS)
                    state_mode = "EXPLORE"
                    hold_timer = 0.0

            # C1-плавность
            current_cmd += (target_cmd - current_cmd) * (dt / 0.30)

            self.shm['target_idx'].value = target_idx
            modes_dict = {"EXPLORE": 0, "HOLD": 1, "SWITCH": 2}
            self.shm['state_mode'].value = modes_dict.get(state_mode, 0)
            self.shm['hold_timer'].value = float(hold_timer)

            # --- ГЕНЕРАЦИЯ LSL СИГНАЛОВ ---
            theta_phase = 2.0 * math.pi * 6.0 * t_vec
            gamma_phase = 2.0 * math.pi * 55.0 * t_vec
            env_nucleus = (np.clip(np.cos(theta_phase), 0, 1) ** 2)
            noise = np.random.normal(0, 0.015, (NUM_CHANNELS, len(t_vec)))

            for node_i in range(NUM_DEVICES):
                cmd_lx, cmd_ly, cmd_rx, cmd_ry = current_cmd[node_i]

                spatial_phase = (COORDS_X * (cmd_lx * 0.35) + COORDS_Y * (cmd_ly * 0.35))[:, None]
                curl_mod = np.where(IS_CORE, -cmd_rx * 0.7, cmd_rx * 0.7)[:, None]
                pwr_mod = (1.0 + cmd_ry * 0.4)

                raw_sig = 10.0 * np.sin(theta_phase + curl_mod * 0.15) + \
                          pwr_mod * 4.5 * env_nucleus * np.sin(gamma_phase + spatial_phase + curl_mod) + noise

                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)

class Synthetic16DDiffusionAgent:
    def __init__(self):
        self.shm = {
            'is_running': mp.Value('b', True),
            'honest_match': mp.Value('d', 0.0),
            'target_idx': mp.Value('i', 0),
            'state_mode': mp.Value('i', 0),
            'hold_timer': mp.Value('d', 0.0)
        }
        self.process = AgentDiffusionProcess(self.shm)
        self.process.start()

    def update_feedback(self, match_score):
        self.shm['honest_match'].value = float(match_score)

    def get_status(self):
        t_idx = self.shm['target_idx'].value
        modes = ["EXPLORE (Наведение)", "HOLD (Удержание)", "SWITCH (Фазовый срыв)"]
        timer = self.shm['hold_timer'].value
        return t_idx, DIFFUSION_TARGETS[t_idx], modes[self.shm['state_mode'].value], timer

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
