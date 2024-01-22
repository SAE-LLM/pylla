__version__ = '1.0'

from .Llama2 import llama2_generator
from .StableDiffusion.automatic_1111 import StableDiffusionAI, TextToImageOptions
from .StableDiffusion.utils import ModelDownloader

def download_model() -> None:
    ModelDownloader.download_model()

def text_to_image(prompt: str, output_path: str , num_inference_steps: int, width: int, height: int) -> None:
    stable_diffusion_instance = StableDiffusionAI()
    options = TextToImageOptions(text=prompt, output_path=output_path , num_inference_steps=num_inference_steps, width=width, height=height)
    stable_diffusion_instance.text_to_image(options)
