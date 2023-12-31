from diffusers import DiffusionPipeline, AutoencoderKL
import torch

def load_diffusion_pipeline():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, torch_dtype=torch.float32, use_safetensors=False, gpu_layers=0
    )
    return pipe

def generate_tintin_img(prompt, num_inference_steps=20):
    pipeline = load_diffusion_pipeline()
    return pipeline(prompt=prompt, num_inference_steps=num_inference_steps).images[0]
