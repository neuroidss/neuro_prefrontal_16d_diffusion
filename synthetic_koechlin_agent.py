#!/usr/bin/env python3
"""
🧠 NEUROCANVAS: CLOSED-LOOP 2D PREFRONTAL AGENT (v73.0)
- Четкое непрерывное наведение на 4 эталона.
- Высокая скорость сходимости к цели (0.12 шаг за фрейм).
"""

import time
import math
import multiprocessing as mp
import numpy as np
from pylsl import StreamInfo, StreamOutlet

FS = 250.0
CHUNK_SIZE = 10
NUM_CHANNELS = 16

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
RADII = np.hypot(COORDS_X, COORDS_Y)
IS_CORE = RADII < 8.0

THETA_GAMMA_TARGETS = [
    {"name": "🏔️ ГОРА (MOUNTAIN)",       "g": 0.0, "s": 0.0, "desc": "F3=0.0 (Природа) | F4=0.0 (День/Свет)"},
    {"name": "🏰 ЗАМОК (CASTLE)",         "g": 1.0, "s": 0.0, "desc": "F3=1.0 (Камень)  | F4=0.0 (День/Свет)"},
    {"name": "🌊 ОКЕАН (OCEAN)",          "g": 0.0, "s": 1.0, "desc": "F3=0.0 (Вода)    | F4=1.0 (Ночь/Шторм)"},
    {"name": "🏙️ НЕБОСКРЕБ (SKYSCRAPER)", "g": 1.0, "s": 1.0, "desc": "F3=1.0 (Стекло)   | F4=1.0 (Ночь/Огни)"}
]

class PrefrontalThetaGammaProcess(mp.Process):
    def __init__(self, shm_dict):
        super().__init__()
        self.daemon = True
        self.shm = shm_dict

    def run(self):
        outlets = {
            "AFz": StreamOutlet(StreamInfo("FreeEEG_AFZ", 'EEG', NUM_CHANNELS, FS, 'float32', 'uid_afz')),
            "F3":  StreamOutlet(StreamInfo("FreeEEG_F3",  'EEG', NUM_CHANNELS, FS, 'float32', 'uid_f3')),
            "F4":  StreamOutlet(StreamInfo("FreeEEG_F4",  'EEG', NUM_CHANNELS, FS, 'float32', 'uid_f4')),
            "Fpz": StreamOutlet(StreamInfo("FreeEEG_FPZ", 'EEG', NUM_CHANNELS, FS, 'float32', 'uid_fpz'))
        }

        start_time = time.time()
        target_idx = 0
        cur_g, cur_s = 0.0, 0.0
        
        state_mode = "EXPLORE"
        hold_timer = 0.0
        last_reset_time = 0.0

        print("🧠 [THETA-GAMMA AGENT] Запущен двухполушарный генератор (v73.0)...")

        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            tgt = THETA_GAMMA_TARGETS[target_idx]
            honest_match = self.shm['honest_match'].value

            # Быстрый градиентный спуск к цели
            if state_mode == "EXPLORE":
                cur_g += (tgt["g"] - cur_g) * 0.12
                cur_s += (tgt["s"] - cur_s) * 0.12

                # Порог захвата 78%
                if honest_match > 0.78:
                    state_mode = "HOLD"
                    hold_timer = 0.0
                    print(f"🎯 [AGENT] Цель захвачена: {tgt['name']} (Match: {honest_match*100:.1f}%)")

            elif state_mode == "HOLD":
                hold_timer += dt
                # Идеальная фиксация в точке цели
                cur_g += (tgt["g"] - cur_g) * 0.2
                cur_s += (tgt["s"] - cur_s) * 0.2
                if hold_timer > 5.0:
                    state_mode = "SWITCH"

            elif state_mode == "SWITCH":
                target_idx = (target_idx + 1) % len(THETA_GAMMA_TARGETS)
                last_reset_time = t_now
                state_mode = "EXPLORE"
                print(f"🔄 [AGENT] Phase Reset Fpz -> Новая цель: {THETA_GAMMA_TARGETS[target_idx]['name']}")

            self.shm['target_idx'].value = target_idx
            self.shm['state_mode'].value = 0 if state_mode == "EXPLORE" else (1 if state_mode == "HOLD" else 2)
            self.shm['hold_timer'].value = float(hold_timer)
            self.shm['agent_g'].value = float(cur_g)
            self.shm['agent_s'].value = float(cur_s)

            theta_phase = 2.0 * math.pi * 6.0 * t_vec
            gamma_phase = 2.0 * math.pi * 60.0 * t_vec

            burst_env = 0.6 + 0.4 * math.sin(2.0 * math.pi * 1.5 * t_now)
            env_nucleus = (np.clip(np.cos(theta_phase), 0, 1) ** 2) * burst_env
            env_pings = (np.clip(np.cos(theta_phase + math.pi), 0, 1) ** 4) * (1.2 - burst_env * 0.4)

            signals = {}
            noise = np.random.normal(0, 0.02, (NUM_CHANNELS, len(t_vec)))

            # 1. AFz: Master Clock
            afz_sig = np.zeros((NUM_CHANNELS, len(t_vec)))
            core_gain = 3.2 if state_mode == "HOLD" else 1.2
            for i in range(NUM_CHANNELS):
                gain = (core_gain + 1.0) if IS_CORE[i] else 0.6
                afz_sig[i, :] = 10.0 * np.sin(theta_phase) + gain * env_nucleus * np.sin(gamma_phase)
            signals['AFz'] = afz_sig + noise

            # 2. F3: Архитектура g
            f3_sig = np.zeros((NUM_CHANNELS, len(t_vec)))
            for i in range(NUM_CHANNELS):
                amp = (1.0 + cur_g * 3.8) if not IS_CORE[i] else (4.8 - cur_g * 3.8)
                sp_f3 = cur_g * math.pi * (COORDS_X[i] / 10.0)
                f3_sig[i, :] = 10.0 * np.sin(theta_phase) + amp * env_nucleus * np.sin(gamma_phase + sp_f3)
            signals['F3'] = f3_sig + noise

            # 3. F4: Природа s
            f4_sig = np.zeros((NUM_CHANNELS, len(t_vec)))
            for i in range(NUM_CHANNELS):
                amp = (1.0 + cur_s * 3.8) if not IS_CORE[i] else (4.8 - cur_s * 3.8)
                sp_f4 = cur_s * math.pi * (COORDS_Y[i] / 10.0)
                f4_sig[i, :] = 10.0 * np.sin(theta_phase) + amp * env_nucleus * np.sin(gamma_phase + sp_f4)
            signals['F4'] = f4_sig + noise

            # 4. Fpz: Phase Reset
            fpz_sig = np.zeros((NUM_CHANNELS, len(t_vec)))
            reset_elapsed = t_now - last_reset_time
            is_resetting = reset_elapsed < 0.25
            
            fpz_theta = theta_phase + (math.pi if is_resetting else 0.0)
            fpz_speed = 3.8 if is_resetting else (0.10 if state_mode == "HOLD" else 1.2)
            
            for i in range(NUM_CHANNELS):
                sp_fpz = fpz_speed * ((COORDS_X[i] + COORDS_Y[i]) / 10.0)
                fpz_sig[i, :] = 10.0 * np.sin(fpz_theta) + 2.5 * env_pings * np.sin(gamma_phase + sp_fpz)
            signals['Fpz'] = fpz_sig + noise

            outlets['AFz'].push_chunk(signals['AFz'].T.tolist())
            outlets['F3'].push_chunk(signals['F3'].T.tolist())
            outlets['F4'].push_chunk(signals['F4'].T.tolist())
            outlets['Fpz'].push_chunk(signals['Fpz'].T.tolist())

            time.sleep(dt)

class PrefrontalHighDimAgent:
    def __init__(self):
        self.shm = {
            'is_running': mp.Value('b', True),
            'honest_match': mp.Value('d', 0.0),
            'target_idx': mp.Value('i', 0),
            'state_mode': mp.Value('i', 0),
            'hold_timer': mp.Value('d', 0.0),
            'agent_g': mp.Value('d', 0.0),
            'agent_s': mp.Value('d', 0.0)
        }
        self.process = PrefrontalThetaGammaProcess(self.shm)
        self.process.start()

    def update_feedback(self, honest_match):
        self.shm['honest_match'].value = float(honest_match)

    def get_target_info(self):
        t_idx = self.shm['target_idx'].value
        modes = ["EXPLORE (Поиск)", "HOLD (Удержание)", "SWITCH (Сброс)"]
        timer = self.shm['hold_timer'].value
        ag_g = self.shm['agent_g'].value
        ag_s = self.shm['agent_s'].value
        return t_idx, THETA_GAMMA_TARGETS[t_idx], modes[self.shm['state_mode'].value], timer, ag_g, ag_s

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
