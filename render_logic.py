# render_logic.py
import torch, cv2, numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, LCMScheduler, AutoencoderTiny

GW, GH = 512, 384

class NeuroRender:
    def __init__(self, mode="lcm", compile_unet=False, remote_conn=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.mode = mode.lower()
        self.remote_conn = remote_conn
        
        if self.remote_conn is not None:
            print("[NeuroRender] Active in REMOTE CLIENT mode. Proxies requests to Server.")
            self.latent_dim = 768
            return
        
        print(f"[NeuroRender] Initializing {self.mode.upper()} pipeline on {self.device}...")
        if self.mode == "turbo":
            self.pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=self.dtype, variant="fp16").to(self.device)
        else:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=self.dtype).to(self.device)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.safety_checker = None
        self.pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        try:
            self.pipe.enable_attention_slicing()
        except Exception as e:
            pass
            
        self.latent_dim = self.pipe.text_encoder.config.hidden_size
        
        if compile_unet and self.device == 'cuda':
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[!] Compilation error: {e}")
        self._warmup()
        print("[NeuroRender] Ready.")

    def _warmup(self):
        dummy_image = Image.fromarray(np.zeros((GH, GW, 3), dtype=np.uint8))
        with torch.no_grad():
            for _ in range(2):
                self.generate(prompt="warmup", image=dummy_image, strength=1.0)

    def encode_prompt(self, prompt_text):
        if self.remote_conn is not None:
            self.remote_conn.send({'cmd': 'encode_prompt', 'text': prompt_text})
            arr = self.remote_conn.recv()
            return torch.tensor(arr, dtype=self.dtype, device=self.device)
            
        with torch.no_grad():
            return self.pipe.text_encoder(self.pipe.tokenizer(
                prompt_text, return_tensors="pt", padding="max_length", 
                max_length=self.pipe.tokenizer.model_max_length, truncation=True
            ).input_ids.to(self.device))[0]

    def generate(self, **kwargs):
        """Полностью универсальный метод. Проксирует любые **kwargs."""
        if self.remote_conn is not None:
            payload = {'cmd': 'generate'}
            
            if 'image' in kwargs and kwargs['image'] is not None:
                from io import BytesIO
                buf = BytesIO()
                kwargs['image'].save(buf, format='JPEG')
                payload['image_bytes'] = buf.getvalue()
                del kwargs['image']
                
            if 'prompt_embeds' in kwargs and kwargs['prompt_embeds'] is not None:
                payload['prompt_embeds'] = kwargs['prompt_embeds'].cpu().numpy()
                del kwargs['prompt_embeds']
                
            payload.update(kwargs)
            self.remote_conn.send(payload)
            
            from io import BytesIO
            resp_bytes = self.remote_conn.recv()
            if isinstance(resp_bytes, dict) and 'error' in resp_bytes:
                raise RuntimeError(resp_bytes['error'])
            return Image.open(BytesIO(resp_bytes))

        # SERVER-SIDE EXECUTION
        if self.mode == "turbo":
            kwargs.setdefault("strength", 0.5)
            kwargs.setdefault("num_inference_steps", 2)
            kwargs.setdefault("guidance_scale", 0.0)
        else:
            kwargs.setdefault("strength", 0.5)
            kwargs.setdefault("num_inference_steps", 4)
            kwargs.setdefault("guidance_scale", 1.0)
            
        return self.pipe(**kwargs).images[0]
