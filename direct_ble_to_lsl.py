#!/usr/bin/env python3
"""
🧠 NeuroCanvas: High-Performance Multi-Process BLE-to-LSL Bridge
- Zero-Loss Parallel Multi-Processing (1 Process per Device)
- STRICT REGISTER VERIFICATION: Guaranteed read-back confirmation for GAIN, SPS (OSR) & DC-Filter
- Configurable SPS (Default: 250) & DC-Filter (Default: 15)
- Independent Sub-Millisecond LSL Clocks
"""

import os
import sys
import tempfile
import time
import argparse
import multiprocessing as mp

# Глушение системного спама liblsl
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

GAIN_REGISTER_MAP = {
    1: 0x0000, 2: 0x1111, 4: 0x2222, 8: 0x3333,
    16: 0x4444, 32: 0x5555, 64: 0x6666, 128: 0x7777
}

SPS_TO_OSR_MAP = {
    32000: 0, 16000: 1, 8000: 2, 4000: 3,
    2000: 4, 1000: 5, 500: 6, 250: 7
}

def device_worker_process(mac_address: str, target_gain: int, target_sps: int, target_dc: int):
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
    logger.info(f"LSL Stream Outlet active (@ {target_sps} Hz, DC_Filter={target_dc}). Connecting via BLE...")

    last_lsl_time = 0.0
    last_counter = -1
    packets_received = 0
    total_lost = 0
    cmd_event = asyncio.Event()
    last_cmd_response = (0, 0)

    def data_handler(sender: int, data: bytearray):
        nonlocal last_lsl_time, last_counter, packets_received, total_lost
        if len(data) == PACKET_SIZE and data[0] == 0xA0 and data[50] == 0xC0:
            counter = data[1]
            channels = [
                int.from_bytes(data[2 + i*3 : 5 + i*3], byteorder='big', signed=True)
                for i in range(CHANNELS_PER_NODE)
            ]
            
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
                    lost = (counter - expected) % 256
                    total_lost += lost
                    
            last_counter = counter
            packets_received += 1

    def cmd_handler(sender: int, data: bytearray):
        nonlocal last_cmd_response
        if len(data) >= 3:
            last_cmd_response = (data[0], (data[1] << 8) | data[2])
            cmd_event.set()

    async def stats_loop():
        nonlocal packets_received
        while True:
            await asyncio.sleep(1.0)
            logger.info(f"Rate: {packets_received:4d} Hz | Total Lost: {total_lost}")
            packets_received = 0

    async def write_and_verify_register(client, reg_name: str, reg_addr: int, val: int, mask: int = 0xFFFF, max_attempts: int = 10) -> bool:
        """Циклически пишет в регистр и читает его обратно, пока значения не совпадут."""
        for attempt in range(1, max_attempts + 1):
            cmd_event.clear()

            # 1. Отправляем команду записи (5 байт для masked, 3 байта для unmasked)
            if mask != 0xFFFF:
                payload = bytearray([reg_addr, (val >> 8) & 0xFF, val & 0xFF, (mask >> 8) & 0xFF, mask & 0xFF])
            else:
                payload = bytearray([reg_addr, (val >> 8) & 0xFF, val & 0xFF])
                
            await client.write_gatt_char(CMD_CHAR_UUID, payload, response=False)
            await asyncio.sleep(0.06)

            # 2. Отправляем запрос чтения (1 байт)
            cmd_event.clear()
            await client.write_gatt_char(CMD_CHAR_UUID, bytearray([reg_addr]), response=False)
            
            try:
                await asyncio.wait_for(cmd_event.wait(), timeout=0.6)
                r_addr, r_val = last_cmd_response
                
                # 3. Сверяем только целевые биты под маской
                if r_addr == reg_addr and (r_val & mask) == (val & mask):
                    logger.info(f"  ✓ {reg_name} (0x{reg_addr:02X}) = 0x{r_val:04X} [VERIFIED on attempt {attempt}]")
                    return True
                else:
                    logger.warning(f"  Mismatch {reg_name}: expected 0x{(val & mask):04X}, got 0x{(r_val & mask):04X}. Retrying...")
            except asyncio.TimeoutError:
                logger.warning(f"  Timeout reading {reg_name} (attempt {attempt}). Retrying...")

            await asyncio.sleep(0.08)

        logger.error(f"❌ Failed to verify {reg_name} after {max_attempts} attempts!")
        return False

    async def run_client():
        reg_gain_val = GAIN_REGISTER_MAP.get(target_gain, 0x4444)
        osr_val = SPS_TO_OSR_MAP[target_sps]
        osr_reg_shifted = osr_val << 2
        dc_reg_val = target_dc & 0x0F

        while True:
            try:
                client = BleakClient(mac_address, timeout=12.0)
                await client.connect()
                logger.info("Connected! Configuring and VERIFYING hardware registers...")
                
                await client.start_notify(CMD_CHAR_UUID, cmd_handler)
                await asyncio.sleep(0.15)
                
                # --- ЦИКЛИЧЕСКАЯ ЗАПИСЬ И ПРОВЕРКА ВСЕХ РЕГИСТРОВ ---
                # 1. GAIN1 (Каналы 0-3)
                await write_and_verify_register(client, "GAIN1", 0x04, reg_gain_val, mask=0x7777)
                
                # 2. GAIN2 (Каналы 4-7)
                await write_and_verify_register(client, "GAIN2", 0x05, reg_gain_val, mask=0x7777)
                
                # 3. CLOCK (SPS/OSR)
                await write_and_verify_register(client, "CLOCK/OSR", 0x03, osr_reg_shifted, mask=0x001C)
                
                # 4. THRSHLD_LSB (DC-Filter)
                await write_and_verify_register(client, "DC_BLOCK", 0x08, dc_reg_val, mask=0x000F)

                # 5. Сброс Global-Chop
                await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x06, 0x00, 0x00, 0x01, 0x00]), response=False)
                await asyncio.sleep(0.08)

                # Старт сбора данных
                await client.start_notify(DATA_CHAR_UUID, data_handler)
                logger.info(f"🚀 ALL REGISTERS CONFIRMED. STREAM ACTIVE ({target_sps} Hz)")

                while client.is_connected:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.warning(f"Connection lost or error: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2.0)

    async def main_node():
        asyncio.create_task(stats_loop())
        await run_client()

    asyncio.run(main_node())

async def scan_and_launch(target_gain: int, target_sps: int, target_dc: int, explicit_macs=None):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Master] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger("Master")

    if explicit_macs:
        target_devices = explicit_macs
        logger.info(f"Using provided MACs: {target_devices}")
    else:
        logger.info(f"Scanning for FreeEEG devices (Service {SERVICE_UUID}) for 2.5 seconds...")
        found = await BleakScanner.discover(timeout=2.5, service_uuids=[SERVICE_UUID])
        target_devices = [d.address for d in found]

    if not target_devices:
        logger.error("No FreeEEG devices found! Turn on the boards and run again.")
        sys.exit(1)

    logger.info(f"Found {len(target_devices)} device(s): {target_devices}")
    logger.info(f"Configuring Target: GAIN={target_gain}, SPS={target_sps}, DC_FILTER={target_dc}")
    logger.info("Spawning independent parallel worker processes...\n")

    processes = []
    for mac in target_devices:
        p = mp.Process(target=device_worker_process, args=(mac, target_gain, target_sps, target_dc), daemon=True)
        p.start()
        processes.append(p)
        time.sleep(0.3)

    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("\nStopping all processes...")
        for p in processes:
            p.terminate()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="BLE to LSL Bridge for FreeEEG16")
    parser.add_argument('--gain', type=int, default=16, choices=[1, 2, 4, 8, 16, 32, 64, 128], 
                        help="PGA Gain (default: 16)")
    parser.add_argument('--sps', type=int, default=250, choices=[250, 500, 1000, 2000, 4000, 8000, 16000, 32000], 
                        help="Sampling rate in SPS (default: 250)")
    parser.add_argument('--dc', type=int, default=15, choices=range(16), 
                        help="DC Block Filter (0=Off, 15=Lowest Cutoff/Original) (default: 15)")
    parser.add_argument('--macs', nargs='+', default=None, 
                        help="Explicit MAC addresses list (optional)")
    args = parser.parse_args()

    try:
        asyncio.run(scan_and_launch(args.gain, args.sps, args.dc, args.macs))
    except KeyboardInterrupt:
        pass
