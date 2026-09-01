#!/usr/bin/env python3
import os, sys, time
from multiprocessing.connection import Listener
from PIL import Image
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_logic import NeuroRender

def main():
    print("🧠 STARTING FAST NEURAL BRAIN SERVER (CUDA BACKEND) 🧠")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    render = NeuroRender(mode="lcm")
    
    listener = Listener(('localhost', 6000), authkey=b'brain')
    print("🧠 Server is ACTIVE. Listening on localhost:6000...")
    
    while True:
        try:
            conn = listener.accept()
            print("[SERVER] Client connected via Direct Memory IPC!")
            while True:
                try:
                    msg = conn.recv()
                    if msg.get('cmd') == 'generate':
                        image_np = msg.pop('image_np', None)
                        embeds_np = msg.pop('prompt_embeds', None)
                        
                        kwargs = msg.copy()
                        del kwargs['cmd']
                        
                        if embeds_np is not None:
                            kwargs['prompt_embeds'] = torch.tensor(embeds_np, dtype=render.dtype, device=render.device)
                        if image_np is not None:
                            kwargs['image'] = Image.fromarray(image_np).convert("RGB")
                            
                        # render_logic.py сам защитит CFG и Steps!
                        out_img = render.generate(**kwargs)
                        conn.send(np.array(out_img, dtype=np.uint8))
                        
                except EOFError:
                    print("[SERVER] Client disconnected.")
                    break
                except Exception as e:
                    try: conn.send({'error': str(e)})
                    except: break
        except Exception as e:
            time.sleep(1)

if __name__ == "__main__":
    main()
