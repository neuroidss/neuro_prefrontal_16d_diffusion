# brain_server.py
#!/usr/bin/env python3
import os, sys, time
from multiprocessing.connection import Listener
from io import BytesIO
from PIL import Image
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_logic import NeuroRender
from vla_jepa_wrapper import VLA_JEPA_Wrapper

def main():
    print("🧠 STARTING FROZEN NEURAL BRAIN SERVER (CUDA BACKEND) 🧠")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading heavy generative models (this takes a while, but only ONCE)...")
    render = NeuroRender(mode="lcm")
    vla_model = VLA_JEPA_Wrapper(render=render)
    
    vocab_atlas_cache = None
    
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'brain')
    print("🧠 Server is ACTIVE and FROZEN. It will never need restarts.")
    
    while True:
        try:
            conn = listener.accept()
            print("[SERVER] Client connected!")
            while True:
                try:
                    msg = conn.recv()
                    cmd = msg.get('cmd')
                    
                    if cmd == 'get_vocab_atlas':
                        if vocab_atlas_cache is None:
                            print("[SERVER] Building Base Atlas...")
                            tokenizer = render.pipe.tokenizer
                            text_model = render.pipe.text_encoder
                            raw_embeds = text_model.get_input_embeddings().weight.detach().to(torch.float32)
                            vocab_size = raw_embeds.shape[0]
                            embed_matrix = raw_embeds / (torch.norm(raw_embeds, dim=-1, keepdim=True) + 1e-7)
                            
                            _, _, V_pca = torch.pca_lowrank(embed_matrix, q=2)
                            coords_2d = torch.matmul(embed_matrix, V_pca[:, :2])
                            max_r = torch.max(torch.norm(coords_2d, dim=-1))
                            coords_2d = coords_2d / (max_r + 1e-6)

                            vocab_atlas_cache = {
                                'embed_matrix': embed_matrix.cpu().numpy(),
                                'coords_2d': coords_2d.cpu().numpy(),
                                'V_pca': V_pca[:, :2].cpu().numpy(),
                                'vocab_size': vocab_size
                            }
                        conn.send(vocab_atlas_cache)

                    elif cmd == 'generate':
                        # ПОЛНАЯ РАЗМОРОЗКА: Сервер берет словарь как есть
                        img_bytes = msg.pop('image_bytes', None)
                        embeds_np = msg.pop('prompt_embeds', None)
                        
                        kwargs = msg.copy()
                        del kwargs['cmd']
                        
                        if embeds_np is not None:
                            kwargs['prompt_embeds'] = torch.tensor(embeds_np, dtype=render.dtype, device=render.device)
                        if img_bytes is not None:
                            kwargs['image'] = Image.open(BytesIO(img_bytes)).convert("RGB")
                            
                        # Проксируем все параметры напрямую в Diffusers
                        out_img = render.generate(**kwargs)
                        
                        buf = BytesIO()
                        out_img.save(buf, format='JPEG', quality=85)
                        conn.send(buf.getvalue())
                        
                except EOFError:
                    print("[SERVER] Client disconnected.")
                    break
                except Exception as e:
                    print(f"[SERVER] Error handling request: {e}")
                    try:
                        conn.send({'error': str(e)})
                    except:
                        break
        except Exception as e:
            print(f"[SERVER] Connection error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
