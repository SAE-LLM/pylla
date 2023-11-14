__version__ = '1.0'

from .StableDiffusion.automatic_1111 import StableDiffusionAI, TextToImageOptions, ImageToTextOptions
from .LLAMA2.llama2 import LLAMA2AI

def text_to_image(prompt: str, output_path: str) -> None:
    stable_diffusion_instance = StableDiffusionAI()
    options = TextToImageOptions(text=prompt, output_path=output_path)
    stable_diffusion_instance.text_to_image(options)

def image_to_text(image_path: str, output_path: str) -> str:
    stable_diffusion_instance = StableDiffusionAI()
    options = ImageToTextOptions(image_path=image_path)
    return stable_diffusion_instance.image_to_text(options)
