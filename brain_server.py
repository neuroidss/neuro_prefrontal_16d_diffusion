#!/usr/bin/env python3
import os, sys, time, argparse
from multiprocessing.connection import Listener
from PIL import Image
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_logic import NeuroRender

def main():
    parser = argparse.ArgumentParser(description="NeuroCanvas Fast Brain Server")
    parser.add_argument('--mode', type=str, default="sdxl-turbo", choices=["lcm", "turbo", "sdxl-turbo"],
                        help="Active diffusion pipeline mode")
    args = parser.parse_args()

    print("🧠 STARTING FAST NEURAL BRAIN SERVER (CUDA BACKEND) 🧠")
    
    current_mode = args.mode
    render = NeuroRender(mode=current_mode)
    
    listener = Listener(('localhost', 6000), authkey=b'brain')
    print(f"🧠 Server is ACTIVE in [{current_mode.upper()}] mode. Listening on localhost:6000...")
    
    while True:
        try:
            conn = listener.accept()
            print(f"[SERVER] Client connected via Direct Memory IPC! Active pipeline: {current_mode.upper()}")
            while True:
                try:
                    msg = conn.recv()
                    cmd = msg.get('cmd')
                    
                    if cmd == 'init_mode':
                        req_mode = msg.get('mode', current_mode)
                        if req_mode != current_mode:
                            print(f"[SERVER] Switching pipeline: {current_mode.upper()} -> {req_mode.upper()}...")
                            del render
                            torch.cuda.empty_cache()
                            render = NeuroRender(mode=req_mode)
                            current_mode = req_mode
                        conn.send({'status': 'ok', 'mode': current_mode, 'is_sdxl': render.is_sdxl})
                        continue

                    elif cmd == 'encode_base_prompts':
                        prompts = msg.get('prompts', [])
                        encoded_data = render.encode_prompts_universal(prompts)
                        conn.send(encoded_data)
                        continue
                        
                    elif cmd == 'generate':
                        image_np = msg.pop('image_np', None)
                        embeds_np = msg.pop('prompt_embeds', None)
                        pooled_np = msg.pop('pooled_prompt_embeds', None)
                        
                        kwargs = msg.copy()
                        del kwargs['cmd']
                        
                        if embeds_np is not None:
                            # Восстанавливаем батч-размерность (1, 77, D)
                            t_emb = torch.tensor(embeds_np, dtype=render.dtype, device=render.device)
                            if t_emb.ndim == 2:
                                t_emb = t_emb.unsqueeze(0)
                            kwargs['prompt_embeds'] = t_emb

                        if pooled_np is not None and render.is_sdxl:
                            t_pool = torch.tensor(pooled_np, dtype=render.dtype, device=render.device)
                            if t_pool.ndim == 1:
                                t_pool = t_pool.unsqueeze(0)
                            kwargs['pooled_prompt_embeds'] = t_pool

                        if image_np is not None:
                            kwargs['image'] = Image.fromarray(image_np).convert("RGB")
                            
                        out_img = render.generate(**kwargs)
                        conn.send(np.array(out_img, dtype=np.uint8))
                        
                except EOFError:
                    print("[SERVER] Client disconnected.")
                    break
                except Exception as e:
                    print(f"[SERVER ERROR]: {e}")
                    try: 
                        conn.send({'error': str(e)})
                    except Exception: 
                        break
        except Exception as e:
            time.sleep(1)

if __name__ == "__main__":
    main()
