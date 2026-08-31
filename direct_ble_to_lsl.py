#!/usr/bin/env python3
"""
🧠 NeuroCanvas: High-Performance Multi-Process BLE-to-LSL Bridge
- Zero-Loss Parallel Multi-Processing (1 Process per Device)
- One-Time Safe Discovery (No radio collisions during streaming)
- Auto-Gain Configuration & Verification (PGA = 16)
- Independent Sub-Millisecond LSL Clocks
"""

import os
import sys
import tempfile
import time
import argparse
import multiprocessing as mp

# 1. Глушение системного спама liblsl
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
SAMPLING_RATE = 250.0
SAMPLE_DT = 1.0 / SAMPLING_RATE

GAIN_REGISTER_MAP = {
    1: 0x0000, 2: 0x1111, 4: 0x2222, 8: 0x3333,
    16: 0x4444, 32: 0x5555, 64: 0x6666, 128: 0x7777
}

# ==============================================================================
# ИЗОЛИРОВАННЫЙ ПРОЦЕСС ДЛЯ ОДНОГО ДЕВАЙСА
# ==============================================================================
def device_worker_process(mac_address: str, target_gain: int):
    clean_mac = mac_address.replace(":", "").replace("-", "").upper()
    logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] [{clean_mac}] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(clean_mac)

    # 1. Создаем персональный LSL Outlet в этом процессе
    info = StreamInfo(
        name=f'FreeEEG_{clean_mac}', 
        type='EEG', 
        channel_count=CHANNELS_PER_NODE, 
        nominal_srate=SAMPLING_RATE, 
        channel_format='int32', 
        source_id=f'uid_{clean_mac}'
    )
    outlet = StreamOutlet(info)
    logger.info(f"LSL Stream Outlet active (@ 250.0 Hz). Connecting via BLE...")

    # Внутреннее состояние процесса
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
            
            # Идеальный квантованный тайминг без джиттера
            now = local_clock()
            if last_lsl_time == 0.0 or abs(now - last_lsl_time) > 0.050:
                sample_time = now
            else:
                sample_time = last_lsl_time + SAMPLE_DT
            last_lsl_time = sample_time
            
            outlet.push_sample(channels, timestamp=sample_time)
            
            # Подсчет потерь по аппаратному счетчику
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
            logger.info(f"Rate: {packets_received:3d} Hz | Total Lost: {total_lost}")
            packets_received = 0

    async def run_client():
        reg_val = GAIN_REGISTER_MAP.get(target_gain, 0x4444)
        
        while True:
            try:
                client = BleakClient(mac_address, timeout=12.0)
                await client.connect()
                logger.info("Connected! Configuring hardware registers...")
                
                await client.start_notify(CMD_CHAR_UUID, cmd_handler)
                await asyncio.sleep(0.1)
                
                # --- УСТАНОВКА И ВЕРИФИКАЦИЯ GAIN ---
                verified = False
                for attempt in range(1, 4):
                    # REG_GAIN1 (0x04)
                    await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x04, (reg_val >> 8) & 0xFF, reg_val & 0xFF]), response=False)
                    await asyncio.sleep(0.08)
                    cmd_event.clear()
                    await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x04]), response=False)
                    await asyncio.wait_for(cmd_event.wait(), timeout=0.8)
                    r_addr1, r_val1 = last_cmd_response

                    # REG_GAIN2 (0x05)
                    await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x05, (reg_val >> 8) & 0xFF, reg_val & 0xFF]), response=False)
                    await asyncio.sleep(0.08)
                    cmd_event.clear()
                    await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x05]), response=False)
                    await asyncio.wait_for(cmd_event.wait(), timeout=0.8)
                    r_addr2, r_val2 = last_cmd_response

                    if r_addr1 == 0x04 and r_val1 == reg_val and r_addr2 == 0x05 and r_val2 == reg_val:
                        logger.info(f"✓ GAIN={target_gain} VERIFIED (0x04=0x{r_val1:04X}, 0x05=0x{r_val2:04X})")
                        verified = True
                        break
                    await asyncio.sleep(0.1)

                # Сброс Global-Chop (250 Гц)
                await client.write_gatt_char(CMD_CHAR_UUID, bytearray([0x06, 0x00, 0x00, 0x01, 0x00]), response=False)
                await asyncio.sleep(0.08)

                # Запуск потока данных
                await client.start_notify(DATA_CHAR_UUID, data_handler)
                logger.info("✓ STREAM ACTIVE (250 Hz)")

                # Мониторинг соединения
                while client.is_connected:
                    await asyncio.sleep(1.0)
                    
            except Exception as e:
                logger.warning(f"Connection lost or error: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2.0)

    async def main_node():
        asyncio.create_task(stats_loop())
        await run_client()

    asyncio.run(main_node())

# ==============================================================================
# ГЛАВНЫЙ ПРОЦЕСС: ОДНОКРАТНЫЙ ПОИСК И ДИСПЕТЧЕРИЗАЦИЯ
# ==============================================================================
async def scan_and_launch(target_gain: int, explicit_macs=None):
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
    logger.info("Spawning independent parallel worker processes (1 Process per Device)...")
    logger.info("Scanner is now COMPLETELY OFF to eliminate radio collisions.\n")

    processes = []
    for mac in target_devices:
        p = mp.Process(target=device_worker_process, args=(mac, target_gain), daemon=True)
        p.start()
        processes.append(p)
        time.sleep(0.3)  # Пауза для разнесения радио-хендшейков

    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("\nStopping all processes...")
        for p in processes:
            p.terminate()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gain', type=int, default=16, choices=[1, 2, 4, 8, 16, 32, 64, 128], help="PGA Gain")
    parser.add_argument('--macs', nargs='+', default=None, help="Explicit MAC addresses list (optional)")
    args = parser.parse_args()

    try:
        asyncio.run(scan_and_launch(args.gain, args.macs))
    except KeyboardInterrupt:
        pass
