__version__ = '1.0'

from .Llama2 import llama2_generator
from pylla.StableDiffusion.automatic_1111 import StableDiffusionAI, TextToImageOptions, ImageToImageOptions, \
    InPaintingOptions, DepthToImageOptions
from pylla.StableDiffusion.utils.model_downloader import ModelDownloader



def download_model() -> None:
    ModelDownloader.download_model()


def text_to_image(prompt: str, output_path: str, num_inference_steps: int, width: int, height: int) -> None:
    stable_diffusion_instance = StableDiffusionAI()
    options = TextToImageOptions(prompt=prompt, output_path=output_path, num_inference_steps=num_inference_steps,
                                 width=width, height=height)
    stable_diffusion_instance.text_to_image(options)


def image_to_image(prompt: str, img_url: str, output_path: str, num_inference_steps: int, width: int,
                   height: int) -> None:
    stable_diffusion_instance = StableDiffusionAI()
    options = ImageToImageOptions(prompt=prompt, img_url=img_url, output_path=output_path,
                                  num_inference_steps=num_inference_steps,
                                  width=width, height=height)
    stable_diffusion_instance.image_to_image(options)


def in_painting(prompt: str, img_url: str, mask_url: str, output_path: str, num_inference_steps: int, width: int,
                height: int) -> None:
    stable_diffusion_instance = StableDiffusionAI()
    options = InPaintingOptions(prompt=prompt, img_url=img_url, mask_url=mask_url, output_path=output_path,
                                num_inference_steps=num_inference_steps,
                                width=width, height=height)
    stable_diffusion_instance.in_painting(options)


def depth_to_image(prompt: str, n_prompt: int, img_url: str, output_path: str, num_inference_steps: int, width: int,
                   height: int) -> None:
    stable_diffusion_instance = StableDiffusionAI()
    options = DepthToImageOptions(prompt=prompt, n_prompt=n_prompt, img_url=img_url, output_path=output_path,
                                  num_inference_steps=num_inference_steps,
                                  width=width, height=height)
    stable_diffusion_instance.depth_to_image(options)
