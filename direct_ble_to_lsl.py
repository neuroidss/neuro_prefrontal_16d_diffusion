#!/usr/bin/env python3
"""
🧠 NeuroCanvas: High-Performance Multi-Process BLE-to-LSL Bridge
- 100% IDENTICAL OVERSATURATION CHECK TO RawDiagnostics.tsx:
    * UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6
    * satThreshold = 299000.0 uV
    * spread = round(maxVal - minVal)
    * satStatus[i] = max(abs(minVal), abs(maxVal)) > satThreshold
- ALWAYS-ON DIAGNOSTICS: Prints formatted status per channel every second.
- PER-CHANNEL AUTO-GAIN: Lowers PGA gain ONLY on the saturated channel.
  (Disabled by default, enabled via --auto-gain).
"""

import os
import sys
import tempfile
import time
import argparse
import multiprocessing as mp

cfg_file = os.path.join(tempfile.gettempdir(), "lsl_api.cfg")
with open(cfg_file, "w") as f:
    f.write("[logging]\nlevel = -2\n")
os.environ["LSLAPICFG"] = cfg_file
os.environ["LIBLSL_LOG_LEVEL"] = "-2"

import asyncio
import logging
from bleak import BleakScanner, BleakClient
from pylsl import StreamInfo, StreamOutlet, local_clock

SERVICE_UUID   = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CMD_CHAR_UUID  = "c0de0001-36e1-4688-b7f5-ea07361b26a8"

CHANNELS_PER_NODE = 16
PACKET_SIZE = 51

# Константы перевода и порога 1:1 из BleService.ts и RawDiagnostics.tsx
UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6  # ~0.0357627881 uV / count
SAT_THRESHOLD_UV = 299000.0                 # Ровно 299000 uV из RawDiagnostics.tsx

GAIN_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
GAIN_TO_CODE = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7}
CODE_TO_GAIN = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128}

SPS_TO_OSR_MAP = {
    32000: 0, 16000: 1, 8000: 2, 4000: 3,
    2000: 4, 1000: 5, 500: 6, 250: 7
}

def device_worker_process(mac_address: str, target_gain: int, target_sps: int, target_dc: int, auto_gain: bool = False):
    clean_mac = mac_address.replace(":", "").replace("-", "").upper()
    logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] [{clean_mac}] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(clean_mac)

    sample_dt = 1.0 / target_sps

    info = StreamInfo(
        name=f'FreeEEG_{clean_mac}', 
        type='EEG', 
        channel_count=CHANNELS_PER_NODE, 
        nominal_srate=float(target_sps), 
        channel_format='int32', 
        source_id=f'uid_{clean_mac}'
    )
    outlet = StreamOutlet(info)
    logger.info(f"LSL Outlet active (@ {target_sps} Hz, Initial Gain={target_gain}x, Auto-Gain={auto_gain}). Connecting...")

    last_lsl_time = 0.0
    last_counter = -1
    packets_received = 0
    total_lost = 0
    cmd_event = asyncio.Event()
    last_cmd_response = (0, 0)

    # Текущее аппаратное усиление по каждому каналу
    channel_gains = [target_gain] * CHANNELS_PER_NODE
    last_gain_reduction_time = [0.0] * CHANNELS_PER_NODE

    # Буферы min / max за текущую секунду (в uV)
    sec_min_uv = [float('inf')] * CHANNELS_PER_NODE
    sec_max_uv = [float('-inf')] * CHANNELS_PER_NODE

    gain_change_queue = asyncio.Queue()

    def data_handler(sender: int, data: bytearray):
        nonlocal last_lsl_time, last_counter, packets_received, total_lost
        if len(data) == PACKET_SIZE and data[0] == 0xA0 and data[50] == 0xC0:
            counter = data[1]
            channels = []
            for i in range(CHANNELS_PER_NODE):
                val = int.from_bytes(data[2 + i*3 : 5 + i*3], byteorder='big', signed=True)
                channels.append(val)

                # Перевод в uV строго по формуле BleService.ts
                v_uv = val * UV_SCALE
                if v_uv < sec_min_uv[i]: sec_min_uv[i] = v_uv
                if v_uv > sec_max_uv[i]: sec_max_uv[i] = v_uv

            now = local_clock()
            if last_lsl_time == 0.0 or abs(now - last_lsl_time) > 0.050:
                sample_time = now
            else:
                sample_time = last_lsl_time + sample_dt
            last_lsl_time = sample_time
            
            outlet.push_sample(channels, timestamp=sample_time)
            
            if last_counter != -1:
                expected = (last_counter + 1) % 256
                if counter != expected:
                    total_lost += (counter - expected) % 256
                    
            last_counter = counter
            packets_received += 1

    def cmd_handler(sender: int, data: bytearray):
        nonlocal last_cmd_response
        if len(data) >= 3:
            last_cmd_response = (data[0], (data[1] << 8) | data[2])
            cmd_event.set()

    async def diagnostics_loop():
        nonlocal packets_received
        while True:
            await asyncio.sleep(1.0)
            now_t = time.time()
            sat_channels = []
            max_spread = 0

            # Точный расчет как в RawDiagnostics.tsx: строки 48-51
            for i in range(CHANNELS_PER_NODE):
                min_v = sec_min_uv[i]
                max_v = sec_max_uv[i]
                sec_min_uv[i] = float('inf')
                sec_max_uv[i] = float('-inf')

                if min_v == float('inf') or max_v == float('-inf'):
                    continue

                spread = int(round(max_v - min_v))
                if spread > max_spread: max_spread = spread

                # satStatus[i] = Math.max(Math.abs(minVal), Math.abs(maxVal)) > satThreshold
                is_saturated = max(abs(min_v), abs(max_v)) > SAT_THRESHOLD_UV

                if is_saturated:
                    sat_channels.append((i, spread, int(round(min_v)), int(round(max_v))))
                    if auto_gain and (now_t - last_gain_reduction_time[i] > 2.0):
                        last_gain_reduction_time[i] = now_t
                        gain_change_queue.put_nowait(i)

            if sat_channels:
                diag_strs = [
                    f"CH {ch} SATURATED ({spread} uV, range: [{vmin}..{vmax}])"
                    for ch, spread, vmin, vmax in sat_channels
                ]
                logger.warning(f"⚠️ [RAW DIAGNOSTICS] {', '.join(diag_strs)}")
            else:
                logger.info(f"Rate: {packets_received:4d} Hz | Lost: {total_lost} | All 16 Channels OK (Max spread: {max_spread} uV)")

            packets_received = 0

    async def single_channel_gain_worker(client):
        """Снижает усиление СТРОГО на одном перегруженном канале по маске"""
        while True:
            ch_idx = await gain_change_queue.get()
            cur_gain = channel_gains[ch_idx]
            cur_code = GAIN_TO_CODE.get(cur_gain, 4)

            if cur_code > 0:
                new_code = cur_code - 1
                new_gain = CODE_TO_GAIN[new_code]
                channel_gains[ch_idx] = new_gain

                # ADS131M08: 0x04 для Ch0-3, 0x05 для Ch4-7
                sub_ch = ch_idx % 8
                reg_addr = 0x04 if sub_ch < 4 else 0x05
                ch_in_reg = sub_ch % 4
                shift = ch_in_reg * 4

                val_to_write  = (new_code << shift)
                mask_to_write = (0x0007   << shift)

                logger.info(f"🔧 [AUTO-GAIN] Reducing Gain ONLY on Ch {ch_idx}: {cur_gain}x -> {new_gain}x (Reg 0x{reg_addr:02X}, Mask 0x{mask_to_write:04X})")
                await write_and_verify_register(client, f"Ch{ch_idx}_PGA", reg_addr, val_to_write, mask=mask_to_write)
            else:
                logger.warning(f"⚠️ [AUTO-GAIN] Ch {ch_idx} already at minimum gain (1x)! Check electrode contact.")
            gain_change_queue.task_done()

    async def write_and_verify_register(client, reg_name: str, reg_addr: int, val: int, mask: int = 0xFFFF, max_attempts: int = 5) -> bool:
        for attempt in range(1, max_attempts + 1):
            cmd_event.clear()
            if mask != 0xFFFF:
                payload = bytearray([reg_addr, (val >> 8) & 0xFF, val & 0xFF, (mask >> 8) & 0xFF, mask & 0xFF])
            else:
                payload = bytearray([reg_addr, (val >> 8) & 0xFF, val & 0xFF])
                
            await client.write_gatt_char(CMD_CHAR_UUID, payload, response=False)
            await asyncio.sleep(0.06)

            cmd_event.clear()
            await client.write_gatt_char(CMD_CHAR_UUID, bytearray([reg_addr]), response=False)
            
            try:
                await asyncio.wait_for(cmd_event.wait(), timeout=0.6)
                r_addr, r_val = last_cmd_response
                if r_addr == reg_addr and (r_val & mask) == (val & mask):
                    logger.info(f"  ✓ {reg_name} (0x{reg_addr:02X}) = 0x{r_val:04X} [VERIFIED on attempt {attempt}]")
                    return True
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.08)

        logger.error(f"❌ Failed to verify {reg_name} (0x{reg_addr:02X})")
        return False

    async def run_client():
        initial_code = GAIN_TO_CODE.get(target_gain, 4)
        initial_gain_reg = (initial_code << 12) | (initial_code << 8) | (initial_code << 4) | initial_code
        osr_val = SPS_TO_OSR_MAP[target_sps]
        osr_reg_shifted = osr_val << 2
        dc_reg_val = target_dc & 0x0F

        while True:
            try:
                client = BleakClient(mac_address, timeout=12.0)
                await client.connect()
                logger.info("Connected! Configuring registers...")
                
                await client.start_notify(CMD_CHAR_UUID, cmd_handler)
                await asyncio.sleep(0.15)
                
                # Запись начального гейна и параметров
                await write_and_verify_register(client, "GAIN1", 0x04, initial_gain_reg, mask=0x7777)
                await write_and_verify_register(client, "GAIN2", 0x05, initial_gain_reg, mask=0x7777)
                await write_and_verify_register(client, "CLOCK/OSR", 0x03, osr_reg_shifted, mask=0x001C)
                await write_and_verify_register(client, "DC_BLOCK", 0x08, dc_reg_val, mask=0x000F)
                await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x06, 0x00, 0x00, 0x01, 0x00]), response=False)
                await asyncio.sleep(0.08)

                if auto_gain:
                    asyncio.create_task(single_channel_gain_worker(client))

                await client.start_notify(DATA_CHAR_UUID, data_handler)
                logger.info(f"🚀 STREAM ACTIVE ({target_sps} Hz). RawDiagnostics check running...")

                while client.is_connected:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.warning(f"Connection lost: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2.0)

    async def main_node():
        asyncio.create_task(diagnostics_loop())
        await run_client()

    asyncio.run(main_node())

async def scan_and_launch(target_gain: int, target_sps: int, target_dc: int, auto_gain: bool, explicit_macs=None):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Master] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger("Master")

    if explicit_macs:
        target_devices = explicit_macs
    else:
        logger.info(f"Scanning for FreeEEG devices (Service {SERVICE_UUID})...")
        found = await BleakScanner.discover(timeout=2.5, service_uuids=[SERVICE_UUID])
        target_devices = [d.address for d in found]

    if not target_devices:
        logger.error("No FreeEEG devices found! Check device power.")
        sys.exit(1)

    logger.info(f"Found {len(target_devices)} device(s): {target_devices}")
    logger.info(f"Config: GAIN={target_gain}x, SPS={target_sps}, DC_FILTER={target_dc}, AUTO_GAIN={auto_gain}")

    processes = []
    for mac in target_devices:
        p = mp.Process(target=device_worker_process, args=(mac, target_gain, target_sps, target_dc, auto_gain), daemon=True)
        p.start()
        processes.append(p)
        time.sleep(0.3)

    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="BLE to LSL Bridge for FreeEEG16 with RawDiagnostics Matching")
    parser.add_argument('--gain', type=int, default=16, choices=GAIN_LEVELS, help="Initial PGA Gain (default: 16)")
    parser.add_argument('--sps', type=int, default=250, choices=list(SPS_TO_OSR_MAP.keys()), help="Sampling rate (default: 250)")
    parser.add_argument('--dc', type=int, default=15, choices=range(16), help="DC Block Filter (default: 15)")
    parser.add_argument('--auto-gain', action='store_true', default=False, 
                        help="Auto-reduce gain ONLY on saturated channels (default: False)")
    parser.add_argument('--macs', nargs='+', default=None, help="Explicit MAC addresses")
    args = parser.parse_args()

    try:
        asyncio.run(scan_and_launch(args.gain, args.sps, args.dc, args.auto_gain, args.macs))
    except KeyboardInterrupt:
        pass
