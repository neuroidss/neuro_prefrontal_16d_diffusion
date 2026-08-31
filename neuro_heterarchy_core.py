"""
🧠 NEURO-HETERARCHY CORE (UNIVERSAL 4-NODE PREFRONTAL DSP)
Детерминированное связывание:
- Node 0: AFz (Master Theta Clock)
- Node 1: F3  (Left DLPFC - Syntax / Macro)
- Node 2: F4  (Right DLPFC - Style / Context)
- Node 3: Fpz (Frontopolar - Branching / Null-Space)
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

FS = 250.0
BUF_SIZE = 256
NUM_CHANNELS = 16
NUM_DEVICES = 4
NUM_FREQS = 32
NUM_PAIRS = 120

COORDS_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
I_IDX, J_IDX = np.triu_indices(NUM_CHANNELS, k=1)

DX_PAIR = COORDS_X[J_IDX] - COORDS_X[I_IDX]
DY_PAIR = COORDS_Y[J_IDX] - COORDS_Y[I_IDX]
TQ_MULT = (COORDS_X[I_IDX] * DY_PAIR - COORDS_Y[I_IDX] * DX_PAIR) / 100.0
SCALE_28_120 = 28.0 / 120.0

@dataclass
class NodeState:
    device_id: int
    name: str
    vx: float
    vy: float
    tq: float
    phase_theta: float
    traj_32: np.ndarray       
    iplv_32: np.ndarray       
    
@dataclass
class UniversalFrame:
    nodes: list[NodeState]
    theta_freq: float
    theta_sync: float
    theta_phase: float
    is_real: bool
    num_live: int

class GPU_Daemon_Process(mp.Process):
    def __init__(self, shared_mem):
        super().__init__()
        self.daemon = True
        self.shm = shared_mem

    def run(self):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        freqs = torch.fft.fftfreq(BUF_SIZE, d=1.0/FS).to(DEVICE)
        notch = torch.ones_like(freqs)
        notch[(torch.abs(freqs) >= 48.0) & (torch.abs(freqs) <= 52.0)] = 0.0
        notch = notch.view(1, 1, BUF_SIZE)

        f_beta = (torch.exp(-0.5 * ((freqs - 22.0) / 8.0)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_beta[:, :, freqs < 0] = 0.0

        f_theta = (torch.exp(-0.5 * ((freqs - 6.0) / 1.5)**2) * 2.0).view(1, 1, BUF_SIZE)
        f_theta[:, :, freqs < 0] = 0.0

        gamma_centers = torch.linspace(30.0, 85.0, NUM_FREQS, device=DEVICE).view(1, NUM_FREQS, 1, 1)
        freqs_4d = freqs.view(1, 1, 1, BUF_SIZE)
        gamma_filters = torch.exp(-0.5 * ((freqs_4d - gamma_centers) / 4.5)**2) * 2.0
        gamma_filters[:, :, :, freqs < 0] = 0.0

        slot_angles = (-math.pi + (2.0 * math.pi / NUM_FREQS) * (torch.arange(NUM_FREQS, device=DEVICE) + 0.5)).view(1, NUM_FREQS, 1, 1)

        I_GPU = torch.from_numpy(I_IDX).to(DEVICE, dtype=torch.long)
        J_GPU = torch.from_numpy(J_IDX).to(DEVICE, dtype=torch.long)
        DX_GPU = torch.from_numpy(DX_PAIR).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)
        DY_GPU = torch.from_numpy(DY_PAIR).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)
        TQ_GPU = torch.from_numpy(TQ_MULT).to(DEVICE, dtype=torch.float32).view(1, NUM_PAIRS)

        inlets = [None] * NUM_DEVICES
        connected_uids = set()
        raw_buffers = np.zeros((NUM_DEVICES, NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        raw_buf_gpu = torch.zeros((NUM_DEVICES, NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)

        sh_vx = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        sh_vy = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        sh_tq = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        sh_dev_phase = np.frombuffer(self.shm['dev_phase'].get_obj(), dtype=np.float64)
        sh_iplv = np.frombuffer(self.shm['iplv'].get_obj(), dtype=np.float64).reshape(NUM_DEVICES, NUM_FREQS, NUM_PAIRS)

        last_resolve_time = 0.0

        while self.shm['is_running'].value:
            now = time.time()
            
            # Детерминированное авто-обнаружение потоков LSL
            if (None in inlets) and (now - last_resolve_time > 1.5):
                last_resolve_time = now
                try:
                    streams = resolve_streams(wait_time=0.1)
                    for s in streams:
                        s_uid = s.uid()
                        s_id_str = (s.name() + " " + s.source_id()).upper()
                        
                        if s_uid not in connected_uids:
                            target_slot = None
                            if "AFZ" in s_id_str: target_slot = 0
                            elif "F3" in s_id_str: target_slot = 1
                            elif "F4" in s_id_str: target_slot = 2
                            elif "FPZ" in s_id_str: target_slot = 3
                            else:
                                for slot_i in range(NUM_DEVICES):
                                    if inlets[slot_i] is None:
                                        target_slot = slot_i
                                        break
                                        
                            if target_slot is not None and inlets[target_slot] is None:
                                try:
                                    inlets[target_slot] = StreamInlet(s, max_buflen=1, max_chunklen=BUF_SIZE, recover=True)
                                    connected_uids.add(s_uid)
                                    print(f"✅ [CORE LSL] Привязан Node [{target_slot}] <- {s.name()} ({s.source_id()})")
                                except Exception:
                                    pass
                except Exception:
                    pass

            num_live = sum(1 for inl in inlets if inl is not None)
            is_real = (num_live > 0)
            self.shm['is_real'].value = is_real
            self.shm['num_live'].value = num_live

            if is_real:
                for i in range(NUM_DEVICES):
                    if inlets[i] is not None:
                        try:
                            chunk, _ = inlets[i].pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                            if chunk:
                                arr = np.array(chunk, dtype=np.float32).T
                                n = arr.shape[1]
                                if n >= BUF_SIZE: raw_buffers[i] = arr[:NUM_CHANNELS, -BUF_SIZE:]
                                else:
                                    raw_buffers[i] = np.roll(raw_buffers[i], -n, axis=1)
                                    raw_buffers[i][:, -n:] = arr[:NUM_CHANNELS, :]
                        except Exception:
                            inlets[i] = None
                raw_buf_gpu.copy_(torch.from_numpy(raw_buffers))
            else:
                time.sleep(0.001)
                continue

            with torch.inference_mode():
                centered = raw_buf_gpu - torch.mean(raw_buf_gpu, dim=2, keepdim=True)
                fft_clean = torch.fft.fft(centered, dim=-1) * notch

                # Бета-кинематика
                Z_beta = torch.fft.ifft(fft_clean * f_beta, dim=-1)
                P_beta = Z_beta / (torch.abs(Z_beta) + 1e-12)
                cg_beta = P_beta[:, I_GPU, :] * torch.conj(P_beta[:, J_GPU, :])
                iplv_beta = torch.mean(torch.imag(cg_beta), dim=-1)

                vx = torch.sum(iplv_beta * DX_GPU, dim=-1) * (SCALE_28_120 * 15.0)
                vy = torch.sum(iplv_beta * DY_GPU, dim=-1) * (SCALE_28_120 * 15.0)
                tq = torch.sum(iplv_beta * TQ_GPU, dim=-1) * (SCALE_28_120 * 18.0)

                # Мастер-Тета (вычисляется из Node 0 = AFz)
                Z_theta = torch.fft.ifft(fft_clean * f_theta, dim=-1)
                P_theta = Z_theta / (torch.abs(Z_theta) + 1e-12)
                mean_th_phasors = torch.mean(P_theta, dim=1) # [4, BUF_SIZE]
                phi_theta_all = torch.angle(mean_th_phasors) # [4, BUF_SIZE]
                
                # Тета с AFz (Node 0) как опорный ритм
                self.shm['theta_phase'].value = float(phi_theta_all[0, -1].item())
                self.shm['theta_sync'].value = float(torch.mean(torch.abs(mean_th_phasors[0])).item())

                th_vec = phi_theta_all[0]
                d_phi = (th_vec[1:] - th_vec[:-1] + math.pi) % (2.0 * math.pi) - math.pi
                self.shm['theta_freq'].value = float(np.clip((torch.mean(d_phi) / (2.0 * math.pi) * FS).item(), 3.5, 9.0))

                # Индивидуальная фаза Теты на каждом девайсе на последнем сэмпле
                dev_phases = phi_theta_all[:, -1].cpu().numpy()

                # Гамма-PAC 32 слота
                fft_exp = fft_clean.unsqueeze(1)
                Z_gamma = torch.fft.ifft(fft_exp * gamma_filters, dim=-1)
                P_gamma = Z_gamma / (torch.abs(Z_gamma) + 1e-12)

                p_diff = phi_theta_all[0:1].view(1, 1, 1, BUF_SIZE) - slot_angles
                w = torch.exp(3.2 * torch.cos(p_diff))
                w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

                cg_gamma = P_gamma[:, :, I_GPU, :] * torch.conj(P_gamma[:, :, J_GPU, :])
                psi_field = torch.sum(cg_gamma * w, dim=-1) # [4, 32, 120]
                
                past_anchor = psi_field[:, 0:1, :]
                gamma_120 = torch.imag(psi_field * torch.conj(past_anchor))

                np.copyto(sh_vx, vx.cpu().numpy())
                np.copyto(sh_vy, vy.cpu().numpy())
                np.copyto(sh_tq, tq.cpu().numpy())
                np.copyto(sh_dev_phase, dev_phases)
                np.copyto(sh_iplv, gamma_120.cpu().numpy())

class HeterarchicalBrainEngine:
    def __init__(self):
        self.shm = {
            'is_running': mp.Value(ctypes.c_bool, True),
            'is_real': mp.Value(ctypes.c_bool, False),
            'num_live': mp.Value('i', 0),
            'theta_sync': mp.Value('d', 0.0),
            'theta_freq': mp.Value('d', 6.0),
            'theta_phase': mp.Value('d', 0.0), 
            'vx': mp.Array('d', NUM_DEVICES),
            'vy': mp.Array('d', NUM_DEVICES),
            'tq': mp.Array('d', NUM_DEVICES),
            'dev_phase': mp.Array('d', NUM_DEVICES),
            'iplv': mp.Array('d', NUM_DEVICES * NUM_FREQS * NUM_PAIRS)
        }
        self._vx = np.frombuffer(self.shm['vx'].get_obj(), dtype=np.float64)
        self._vy = np.frombuffer(self.shm['vy'].get_obj(), dtype=np.float64)
        self._tq = np.frombuffer(self.shm['tq'].get_obj(), dtype=np.float64)
        self._dev_phase = np.frombuffer(self.shm['dev_phase'].get_obj(), dtype=np.float64)
        self._iplv = np.frombuffer(self.shm['iplv'].get_obj(), dtype=np.float64).reshape(NUM_DEVICES, NUM_FREQS, NUM_PAIRS)

        self.node_names = ["AFz", "F3", "F4", "Fpz"]
        self.process = GPU_Daemon_Process(self.shm)

    def start(self): 
        self.process.start()
    
    def stop(self):
        self.shm['is_running'].value = False
        self.process.join(timeout=1.0)

    def get_frame(self) -> UniversalFrame:
        nodes = []
        for i in range(NUM_DEVICES):
            nodes.append(NodeState(
                device_id=i,
                name=self.node_names[i],
                vx=self._vx[i], 
                vy=self._vy[i], 
                tq=self._tq[i],
                phase_theta=self._dev_phase[i],
                traj_32=np.zeros((NUM_FREQS, 2)),
                iplv_32=self._iplv[i].copy()
            ))
        return UniversalFrame(
            nodes=nodes,
            theta_freq=self.shm['theta_freq'].value, 
            theta_sync=self.shm['theta_sync'].value,
            theta_phase=self.shm['theta_phase'].value, 
            is_real=self.shm['is_real'].value, 
            num_live=self.shm['num_live'].value
        )
