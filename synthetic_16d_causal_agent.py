#!/usr/bin/env python3
"""
🧠 AUTONOMOUS PREFRONTAL AGENT (Hawkins Goal-Attractor Edition)
- F3/F4 передают чистый аттрактор цели в рабочую память.
- Активный вывод: Поиск -> Дофамин (Успех) -> Скука -> Саккада.
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

TARGETS = [
    {"name": "ГОРА",       "g": 0.0, "s": 0.0},
    {"name": "ЗАМОК",      "g": 1.0, "s": 0.0},
    {"name": "НЕБОСКРЕБ", "g": 1.0, "s": 1.0},
    {"name": "ОКЕАН",      "g": 0.0, "s": 1.0}
]

class AutonomousAgentProcess(mp.Process):
    def __init__(self, shm):
        super().__init__()
        self.daemon = True
        self.shm = shm

    def run(self):
        outlets = []
        for i in range(NUM_DEVICES):
            info = StreamInfo(f'FreeEEG_Node{i}', 'EEG', NUM_CHANNELS, FS, 'float32', f'sim_diff_node_{i}')
            outlets.append(StreamOutlet(info))

        print("🤖 [AUTONOMOUS BRAIN AGENT] Префронтальная кора запущена (250 Hz)...")
        start_time = time.time()
        
        tgt_idx = 1 # Стартуем с Замка
        boredom = 0.0
        frustration = 0.0
        
        while self.shm['is_running'].value:
            dt = CHUNK_SIZE / FS
            t_now = time.time() - start_time
            t_vec = np.linspace(t_now, t_now + dt, CHUNK_SIZE, endpoint=False)

            scr_g = self.shm['screen_g'].value
            scr_s = self.shm['screen_s'].value

            # Семантическое распознавание
            dists = [math.hypot(scr_g - t["g"], scr_s - t["s"]) for t in TARGETS]
            best_idx = int(np.argmin(dists))
            perceived_name = TARGETS[best_idx]["name"]
            
            cur_target = TARGETS[tgt_idx]
            dist_to_goal = dists[tgt_idx]
            satisfaction = float(np.clip(1.0 - (dist_to_goal / 0.55), 0.0, 1.0))

            is_calib = t_now < 16.0
            if is_calib:
                mode = "CALIBRATION"
                tgt_g = scr_g
                tgt_s = scr_s
                state_desc = "Синхронизация базисов с эталонами среды..."
            else:
                tgt_g = cur_target["g"]
                tgt_s = cur_target["s"]

                if satisfaction >= 0.70:
                    mode = f"SATISFIED ({satisfaction*100:.0f}%)"
                    boredom += dt
                    frustration = max(0.0, frustration - dt)
                    state_desc = f"Вижу {perceived_name}! Изучаю. Скука: {int(boredom/4.0*100)}%"
                else:
                    mode = "SEEKING"
                    frustration += dt
                    boredom = max(0.0, boredom - dt)
                    state_desc = f"Ищу {cur_target['name']} (вижу {perceived_name}). Застой: {int(frustration/8.0*100)}%"

                # Смена цели по скуке или застою
                if boredom >= 4.0:
                    tgt_idx = (tgt_idx + 1) % 4
                    boredom = 0.0
                    frustration = 0.0
                    mode = "SACCADE"
                    state_desc = f"Надоело. Переключаюсь на {TARGETS[tgt_idx]['name']}."
                elif frustration >= 8.0:
                    tgt_idx = (tgt_idx + 1) % 4
                    boredom = 0.0
                    frustration = 0.0
                    mode = "EXPLORATION"
                    state_desc = f"Застрял! Срыв на {TARGETS[tgt_idx]['name']}."

            self.shm['agent_mode'].value = mode.encode('utf-8')[:127].ljust(128, b'\x00')
            self.shm['agent_desc'].value = state_desc.encode('utf-8')[:255].ljust(256, b'\x00')
            self.shm['agent_tgt_g'].value = float(tgt_g)
            self.shm['agent_tgt_s'].value = float(tgt_s)
            self.shm['satisfaction'].value = float(satisfaction)
            self.shm['boredom'].value = float(boredom / 4.0)
            self.shm['frustration'].value = float(frustration / 8.0)

            # ЧИСТЫЙ АТТРАКТОР ЦЕЛИ В РАБОЧЕЙ ПАМЯТИ
            target_16d = np.zeros((4, 4), dtype=np.float32)
            target_16d[0, 0] = tgt_g * 2.0 - 1.0 # F3 lx
            target_16d[0, 1] = 0.30
            target_16d[1, 0] = tgt_s * 2.0 - 1.0 # F4 lx
            target_16d[1, 1] = 0.30
            target_16d[3, 2] = min(1.0, max(boredom / 4.0, frustration / 8.0))

            # Синтез LSL
            theta_phase = TWO_PI * 6.0 * t_vec
            gamma_phase = TWO_PI * 55.0 * t_vec
            env_nucleus = (np.clip(np.cos(theta_phase), 0, 1) ** 2)
            noise = np.random.normal(0, 0.008, (NUM_CHANNELS, len(t_vec)))

            for node_i in range(NUM_DEVICES):
                cmd_lx, cmd_ly, cmd_rx, cmd_ry = target_16d[node_i]
                spatial_phase = (COORDS_X * (cmd_lx * 0.4) + COORDS_Y * (cmd_ly * 0.4))[:, None]
                curl_mod = np.where(IS_CORE, -cmd_rx * 0.7, cmd_rx * 0.7)[:, None]
                
                raw_sig = 10.0 * np.sin(theta_phase) + \
                          4.5 * env_nucleus * np.sin(gamma_phase + spatial_phase + curl_mod) + noise
                outlets[node_i].push_chunk(raw_sig.T.tolist())

            time.sleep(dt)

class SyntheticAutonomousAgent:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.shm = {
            'is_running': ctx.Value('b', True),
            'screen_g': ctx.Value('d', 0.5), 
            'screen_s': ctx.Value('d', 0.5),
            'agent_mode': ctx.Array('c', 128),
            'agent_desc': ctx.Array('c', 256),
            'agent_tgt_g': ctx.Value('d', 0.5),
            'agent_tgt_s': ctx.Value('d', 0.5),
            'satisfaction': ctx.Value('d', 0.0),
            'boredom': ctx.Value('d', 0.0),
            'frustration': ctx.Value('d', 0.0)
        }
        self.process = AutonomousAgentProcess(self.shm)
        self.process.start()

    def update_screen_state(self, g, s):
        self.shm['screen_g'].value = float(g)
        self.shm['screen_s'].value = float(s)

    def get_telemetry(self):
        mode = self.shm['agent_mode'].value.decode('utf-8').replace('\x00', '').strip()
        desc = self.shm['agent_desc'].value.decode('utf-8').replace('\x00', '').strip()
        tg = self.shm['agent_tgt_g'].value
        ts = self.shm['agent_tgt_s'].value
        sat = self.shm['satisfaction'].value
        bor = self.shm['boredom'].value
        fru = self.shm['frustration'].value
        return mode, desc, tg, ts, sat, bor, fru

    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)
