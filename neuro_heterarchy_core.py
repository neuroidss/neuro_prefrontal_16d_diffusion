#!/usr/bin/env python3
"""
🧠 NEURO-HETERARCHY CORE v100.0 (DUAL-BAND 500 Hz HAL)
- 500 SPS: Найквист = 250 Гц.
- Контур A (30–85 Гц): 32 фильтра Low/Fast Gamma для семантики концептов (Colgin 2009).
- Контур B (100–200 Гц): 32 фильтра Cortical Ripples для рекурсии и связывания (Dickey 2022).
- Контур C (1.5 / 6.0 Гц): Дельта-Тета пейсмейкер макро-фреймов (Ding 2016, Lakatos 2005).
- Строго знаковый directed iPLV: sin(Δφ) ∈ [-1, +1] (Bruña et al., 2018).
"""

import os
import time
import math
import ctypes
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
import torch
from pylsl import StreamInlet, resolve_streams

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

FS = 500.0
BUF_SIZE = 256  # 512 мс буфер (Δf = 1.95 Гц)
NUM_CHANNELS = 16
NUM_MAX_DEVICES = 4
NUM_SLOTS = 32
NUM_PAIRS = 120

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 26-мм концентрическая геометрия FreeEEG16-alpha2
COORDS_X = np.array([
    10.14,  7.43,  2.75,  2.72, -2.72, -2.75, -7.42, -10.14,
   -10.14, -7.43, -2.75, -2.72,  2.72,  2.75,  7.43,  10.14
], dtype=np.float32)
COORDS_Y = np.array([
    -2.72, -7.43, -4.77, -10.15,-10.14, -4.77, -7.42,  -2.73,
     2.72,  7.43,  4.76,  10.14, 10.15,  4.77,  7.42,   2.71
], dtype=np.float32)
I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)

DX_PAIR = (COORDS_X[J_IDX] - COORDS_X[I_IDX]).astype(np.float32)
DY_PAIR = (COORDS_Y[J_IDX] - COORDS_Y[I_IDX]).astype(np.float32)

DX_GPU = torch.from_numpy(DX_PAIR).to(DEVICE)
DY_GPU = torch.from_numpy(DY_PAIR).to(DEVICE)
I_GPU  = torch.from_numpy(I_IDX).to(DEVICE, dtype=torch.long)
J_GPU  = torch.from_numpy(J_IDX).to(DEVICE, dtype=torch.long)

@dataclass
class Kinematics4D:
    lx: float
    ly: float
    rx: float
    ry: float

@dataclass
class NodeState:
    device_id: int
    name: str
    source_id: str
    is_connected: bool
    phase_theta: float
    phase_delta: float
    beta_stability: float
    disp_xyz: np.ndarray        # [3] displacement для CMP
    pose_matrix: np.ndarray     # [3, 3] SO(3) позы для CMP
    kinematics: Kinematics4D
    iplv_gamma: np.ndarray      # [32, 120] 30–85 Гц (Контент)
    iplv_ripple: np.ndarray     # [32, 120] 100–200 Гц (Вложенность/Рекурсия)

@dataclass
class UniversalFrame:
    nodes: list[NodeState]
    theta_freq: float
    delta_freq: float
    theta_sync: float
    theta_phase: float
    delta_phase: float
    is_real: bool
    num_live: int

class GPU_Daemon_Process(mp.Process):
    def __init__(self, shared_mem):
        super().__init__()
        self.daemon = True
        self.shm = shared_mem

    def run(self):
        freqs = torch.fft.fftfreq(BUF_SIZE, d=1.0/FS).to(DEVICE)
        
        # Режекторные фильтры сетевой наводки
        notch = torch.ones_like(freqs)
        notch[(torch.abs(freqs) >= 48.0) & (torch.abs(freqs) <= 52.0)] = 0.0
        notch[(torch.abs(freqs) >= 98.0) & (torch.abs(freqs) <= 102.0)] = 0.0
        notch = notch.view(1, 1, BUF_SIZE)

        # 1. Пейсмейкеры: Дельта (1.5 Гц) и Тета (6.0 Гц)
        f_delta = (torch.exp(-0.5 * ((freqs - 1.5) / 0.6)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_delta[:, :, freqs < 0] = 0.0

        f_theta = (torch.exp(-0.5 * ((freqs - 6.0) / 1.5)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_theta[:, :, freqs < 0] = 0.0

        f_beta  = (torch.exp(-0.5 * ((freqs - 22.0) / 6.0)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_beta[:, :, freqs < 0] = 0.0

        freqs_4d = freqs.view(1, 1, 1, BUF_SIZE)

        # 2. Фильтрбанк A: Low/Fast Gamma (30.0 - 85.0 Гц) для семантики
        gamma_centers = torch.linspace(30.0, 85.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
        gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / 4.5)**2) * 2.0
        gamma_filters[:, :, :, freqs < 0] = 0.0

        # 3. Фильтрбанк B: Cortical Ripples (100.0 - 200.0 Гц) для рекурсии
        ripple_centers = torch.linspace(100.0, 200.0, NUM_SLOTS, device=DEVICE).view(1, NUM_SLOTS, 1, 1)
        ripple_filters = torch.exp(-0.5 * ((freqs_4d - ripple_centers) / 6.0)**2) * 2.0
        ripple_filters[:, :, :, freqs < 0] = 0.0

        # 32 фазовых слота Теты (от -π до +π)
        slot_angles = (-math.pi + (2.0 * math.pi / NUM_SLOTS) * (torch.arange(NUM_SLOTS, device=DEVICE) + 0.5)).view(1, NUM_SLOTS, 1, 1)

        # jPCA матрица ротации
        M_skew = torch.tensor([
            [0.0, -3.2,  0.5],
            [3.2,  0.0, -2.1],
            [-0.5, 2.1,  0.0]
        ], device=DEVICE, dtype=torch.float32)

        inlets = [None] * NUM_MAX_DEVICES
        stream_names = ["Empty"] * NUM_MAX_DEVICES
        stream_uids = [""] * NUM_MAX_DEVICES
        connected_uids = set()

        raw_buffers = np.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        raw_buf_gpu = torch.zeros((NUM_MAX_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)

        # Связывание Shared Memory
        sh_dev_th_phase = np.frombuffer(self.shm['dev_th_phase'].get_obj(), dtype=np.float64)
        sh_dev_dl_phase = np.frombuffer(self.shm['dev_dl_phase'].get_obj(), dtype=np.float64)
        sh_kinematics   = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        sh_disp         = np.frombuffer(self.shm['disp'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3)
        sh_pose         = np.frombuffer(self.shm['pose'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 9)
        sh_beta_stab    = np.frombuffer(self.shm['beta_stab'].get_obj(), dtype=np.float64)
        sh_iplv_gamma   = np.frombuffer(self.shm['iplv_gamma'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        sh_iplv_ripple  = np.frombuffer(self.shm['iplv_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)

        prev_beta_vecs = torch.zeros((NUM_MAX_DEVICES, 2), device=DEVICE)
        last_th_phase = 0.0
        last_dl_phase = 0.0
        last_time = time.time()
        last_resolve_time = 0.0

        while self.shm['is_running'].value:
            now = time.time()
            dt = max(0.001, min(0.05, now - last_time))
            last_time = now

            if (None in inlets) and (now - last_resolve_time > 1.2):
                last_resolve_time = now
                try:
                    streams = resolve_streams(wait_time=0.1)
                    streams = sorted(streams, key=lambda s: s.name())
                    for s in streams:
                        s_uid = s.uid()
                        if s_uid not in connected_uids and s.channel_count() == NUM_CHANNELS:
                            for slot_i in range(NUM_MAX_DEVICES):
                                if inlets[slot_i] is None:
                                    try:
                                        inlets[slot_i] = StreamInlet(s, max_buflen=1, max_chunklen=BUF_SIZE, recover=True)
                                        connected_uids.add(s_uid)
                                        stream_names[slot_i] = s.name()
                                        stream_uids[slot_i] = s.source_id()
                                        print(f"✅ [CORE HAL 500Hz] Подключен узел [{slot_i}] <- '{s.name()}'")
                                        break
                                    except Exception:
                                        pass
                except Exception:
                    pass

            num_live = sum(1 for inl in inlets if inl is not None)
            is_real = (num_live > 0)
            self.shm['is_real'].value = is_real
            self.shm['num_live'].value = num_live

            if is_real:
                for i in range(NUM_MAX_DEVICES):
                    if inlets[i] is not None:
                        try:
                            chunk, _ = inlets[i].pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                            if chunk:
                                arr = np.array(chunk, dtype=np.float32).T
                                n = arr.shape[1]
                                if n >= BUF_SIZE:
                                    raw_buffers[i] = arr[:NUM_CHANNELS, -BUF_SIZE:]
                                else:
                                    raw_buffers[i] = np.roll(raw_buffers[i], -n, axis=1)
                                    raw_buffers[i][:, -n:] = arr[:NUM_CHANNELS, :]
                        except Exception:
                            if stream_uids[i] in connected_uids:
                                connected_uids.remove(stream_uids[i])
                            inlets[i] = None
                            stream_names[i] = "Empty"
                raw_buf_gpu.copy_(torch.from_numpy(raw_buffers))
            else:
                time.sleep(0.001)
                continue

            with torch.inference_mode():
                centered = raw_buf_gpu - torch.mean(raw_buf_gpu, dim=2, keepdim=True)
                fft_clean = torch.fft.fft(centered, dim=-1) * notch

                first_active = 0
                for idx in range(NUM_MAX_DEVICES):
                    if inlets[idx] is not None:
                        first_active = idx
                        break

                # 1. Тета и Дельта фазовые часы
                Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
                P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
                mean_th_phasors = torch.mean(P_theta, dim=1)
                phi_theta_all = torch.angle(mean_th_phasors)

                cur_th_p = float(phi_theta_all[first_active, -1].item())
                d_th = (cur_th_p - last_th_phase + math.pi) % (2.0 * math.pi) - math.pi
                last_th_phase = cur_th_p
                self.shm['theta_phase'].value = cur_th_p
                self.shm['theta_freq'].value = float(np.clip(abs(d_th / dt) / (2.0 * math.pi), 3.5, 9.0))
                self.shm['theta_sync'].value = float(torch.mean(torch.abs(mean_th_phasors[first_active])).item())

                Z_delta = torch.fft.ifft(fft_clean * f_delta, dim=-1)
                P_delta = Z_delta / (torch.abs(Z_delta) + 1e-12)
                mean_dl_phasor = torch.mean(P_delta, dim=1)
                phi_delta_all = torch.angle(mean_dl_phasor)

                cur_dl_p = float(phi_delta_all[first_active, -1].item())
                d_dl = (cur_dl_p - last_dl_phase + math.pi) % (2.0 * math.pi) - math.pi
                last_dl_phase = cur_dl_p
                self.shm['delta_phase'].value = cur_dl_p
                self.shm['delta_freq'].value = float(np.clip(abs(d_dl / dt) / (2.0 * math.pi), 1.0, 3.5))

                # 2. Бета-стабильность
                Z_beta = torch.fft.ifft(fft_clean * f_beta, dim=-1)
                P_beta = Z_beta / (torch.abs(Z_beta) + 1e-12)
                cg_beta = P_beta[:, I_GPU, :] * torch.conj(P_beta[:, J_GPU, :])
                iplv_beta = torch.mean(torch.imag(cg_beta), dim=-1)

                v_bx = torch.sum(iplv_beta * DX_GPU, dim=-1) / 120.0
                v_by = torch.sum(iplv_beta * DY_GPU, dim=-1) / 120.0
                cur_beta_vecs = torch.stack([v_bx, v_by], dim=-1)

                dot_b = torch.sum(cur_beta_vecs * prev_beta_vecs, dim=-1)
                norm_b = torch.norm(cur_beta_vecs, dim=-1) * torch.norm(prev_beta_vecs, dim=-1) + 1e-6
                beta_stabs = torch.clamp(dot_b / norm_b, -1.0, 1.0)
                prev_beta_vecs.copy_(cur_beta_vecs)

                # Веса слотов по фазе Теты (von Mises)
                p_diff = phi_theta_all[first_active:first_active+1].view(1, 1, 1, BUF_SIZE) - slot_angles
                w = torch.exp(3.2 * torch.cos(p_diff))
                w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

                fft_exp = fft_clean.unsqueeze(1)

                # 3. КОНТУР A: 30–85 Гц Гамма (Семантика концептов)
                Z_gamma = torch.fft.ifft(fft_exp * gamma_filters, dim=-1)
                P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)
                cg_gamma = P_gamma[:, :, I_GPU, :] * torch.conj(P_gamma[:, :, J_GPU, :])
                psi_gamma = torch.sum(cg_gamma * w, dim=-1)
                gamma_120 = torch.imag(psi_gamma * torch.conj(psi_gamma[:, 0:1, :])) # [4, 32, 120]

                # 4. КОНТУР B: 100–200 Гц Рипплы (Структура рекурсии)
                Z_ripple = torch.fft.ifft(fft_exp * ripple_filters, dim=-1)
                P_ripple = Z_ripple / (torch.abs(Z_ripple) + 1e-12)
                cg_ripple = P_ripple[:, :, I_GPU, :] * torch.conj(P_ripple[:, :, J_GPU, :])
                psi_ripple = torch.sum(cg_ripple * w, dim=-1)
                ripple_120 = torch.imag(psi_ripple * torch.conj(psi_ripple[:, 0:1, :])) # [4, 32, 120]

                # 5. Кинематика jPCA
                v_gx = torch.sum(gamma_120[:, 31] * DX_GPU, dim=-1) / 120.0
                v_gy = torch.sum(gamma_120[:, 31] * DY_GPU, dim=-1) / 120.0
                g_vecs = torch.stack([v_gx, v_gy, v_gx * 0.5], dim=-1)

                disp_all = torch.matmul(g_vecs, M_skew.T) * dt * 4.0

                u1 = disp_all / (torch.norm(disp_all, dim=-1, keepdim=True) + 1e-6)
                up_ref = torch.tensor([0.0, 0.0, 1.0], device=DEVICE).view(1, 3).expand(NUM_MAX_DEVICES, 3)
                u2 = torch.cross(u1, up_ref, dim=-1)
                u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + 1e-6)
                u3 = torch.cross(u1, u2, dim=-1)
                pose_matrices = torch.stack([u1, u2, u3], dim=1)

                kinematics_gpu = torch.stack([v_gx, v_gy, beta_stabs, torch.tensor([self.shm['theta_freq'].value / 10.0]*4, device=DEVICE)], dim=-1)

                np.copyto(sh_dev_th_phase, phi_theta_all[:, -1].cpu().numpy())
                np.copyto(sh_dev_dl_phase, phi_delta_all[:, -1].cpu().numpy())
                np.copyto(sh_kinematics, kinematics_gpu.cpu().numpy())
                np.copyto(sh_disp, disp_all.cpu().numpy())
                np.copyto(sh_pose, pose_matrices.contiguous().view(NUM_MAX_DEVICES, 9).cpu().numpy())
                np.copyto(sh_beta_stab, beta_stabs.cpu().numpy())
                np.copyto(sh_iplv_gamma, gamma_120.cpu().numpy())
                np.copyto(sh_iplv_ripple, ripple_120.cpu().numpy())

class HeterarchicalBrainEngine:
    def __init__(self):
        self.shm = {
            'is_running': mp.Value(ctypes.c_bool, True),
            'is_real': mp.Value(ctypes.c_bool, False),
            'num_live': mp.Value('i', 0),
            'theta_sync': mp.Value('d', 0.0),
            'theta_freq': mp.Value('d', 6.0),
            'delta_freq': mp.Value('d', 1.5),
            'theta_phase': mp.Value('d', 0.0),
            'delta_phase': mp.Value('d', 0.0),
            'dev_th_phase': mp.Array('d', NUM_MAX_DEVICES),
            'dev_dl_phase': mp.Array('d', NUM_MAX_DEVICES),
            'kinematics': mp.Array('d', NUM_MAX_DEVICES * 4),
            'disp': mp.Array('d', NUM_MAX_DEVICES * 3),
            'pose': mp.Array('d', NUM_MAX_DEVICES * 9),
            'beta_stab': mp.Array('d', NUM_MAX_DEVICES),
            'iplv_gamma': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * NUM_PAIRS),
            'iplv_ripple': mp.Array('d', NUM_MAX_DEVICES * NUM_SLOTS * NUM_PAIRS)
        }
        self._dev_th_phase = np.frombuffer(self.shm['dev_th_phase'].get_obj(), dtype=np.float64)
        self._dev_dl_phase = np.frombuffer(self.shm['dev_dl_phase'].get_obj(), dtype=np.float64)
        self._kinematics   = np.frombuffer(self.shm['kinematics'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 4)
        self._disp         = np.frombuffer(self.shm['disp'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3)
        self._pose         = np.frombuffer(self.shm['pose'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, 3, 3)
        self._beta_stab    = np.frombuffer(self.shm['beta_stab'].get_obj(), dtype=np.float64)
        self._iplv_gamma   = np.frombuffer(self.shm['iplv_gamma'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        self._iplv_ripple  = np.frombuffer(self.shm['iplv_ripple'].get_obj(), dtype=np.float64).reshape(NUM_MAX_DEVICES, NUM_SLOTS, NUM_PAIRS)
        
        self.process = GPU_Daemon_Process(self.shm)

    def start(self): self.process.start()
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)

    def get_frame(self) -> UniversalFrame:
        nodes = []
        names = ["F3 (Logic/Syntax)", "F4 (Semantics/Style)", "AFz (Rule/Metric)", "Fpz (Meta/Branching)"]
        for i in range(NUM_MAX_DEVICES):
            k = self._kinematics[i]
            nodes.append(NodeState(
                device_id=i,
                name=names[i],
                source_id=f"Slot_{i}",
                is_connected=bool(self._dev_th_phase[i] != 0.0),
                phase_theta=float(self._dev_th_phase[i]),
                phase_delta=float(self._dev_dl_phase[i]),
                beta_stability=float(self._beta_stab[i]),
                disp_xyz=self._disp[i].copy(),
                pose_matrix=self._pose[i].copy(),
                kinematics=Kinematics4D(lx=float(k[0]), ly=float(k[1]), rx=float(k[2]), ry=float(k[3])),
                iplv_gamma=self._iplv_gamma[i].copy(),
                iplv_ripple=self._iplv_ripple[i].copy()
            ))
        return UniversalFrame(
            nodes=nodes,
            theta_freq=self.shm['theta_freq'].value,
            delta_freq=self.shm['delta_freq'].value,
            theta_sync=self.shm['theta_sync'].value,
            theta_phase=self.shm['theta_phase'].value,
            delta_phase=self.shm['delta_phase'].value,
            is_real=self.shm['is_real'].value,
            num_live=self.shm['num_live'].value
        )
