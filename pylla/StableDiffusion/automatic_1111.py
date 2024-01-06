import torch
from diffusers import StableDiffusionPipeline
from dataclasses import dataclass
import pylla.StableDiffusion.utils as utils


@dataclass
class TextToImageOptions:
    prompt: str
    output_path: str
    num_inference_steps: int
    width: int = 256
    height: int = 256

@dataclass
class ImageToImageOptions:
    prompt: str
    img_url: str
    output_path: str
    num_inference_steps: int
    width: int = 256
    height: int = 256

@dataclass
class InPaintingOptions:
    prompt: str
    img_url: str
    mask_url: str
    output_path: str
    num_inference_steps: int
    width: int = 256
    height: int = 256

@dataclass
class DepthToImageOptions:
    prompt: str
    n_prompt: int
    img_url: str
    output_path: str
    num_inference_steps: int
    width: int = 256
    height: int = 256

class StableDiffusionAI:
    def __init__(self, model_path: str = "models/stable-diffusion-2.ckpt") -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            local_files_only=True
        )

    def text_to_image(self, options: TextToImageOptions) -> None:
        print("---Text To Image---")
        print(f"Text: {options.prompt}")
        print(f"Output Image Path: {options.output_path}")  
        prompt = options.prompt
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        image = self.pipe(prompt, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        image.save(options.output_path)

    def image_to_image(self, options: ImageToImageOptions) -> None:
        print("---Image To Image---")
        print(f"Text: {options.prompt}")
        print(f"Image URL: {options.img_url}")
        print(f"Output Image Path: {options.output_path}")
        prompt = options.prompt
        img_url = options.img_url
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        image = utils.get_image(img_url)
        image = self.pipe(prompt, image, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        image.save(options.output_path)

    def in_painting(self, options: InPaintingOptions) -> None:
        print("---In Painting---")
        print(f"Text: {options.prompt}")
        print(f"Image URL: {options.img_url}")
        print(f"Mask URL: {options.mask_url}")
        print(f"Output Image Path: {options.output_path}")
        prompt = options.prompt
        img_url = options.img_url
        mask_url = options.mask_url
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        image = utils.get_image(img_url)
        mask = utils.get_image(mask_url)
        image = self.pipe(prompt=prompt, image=image, mask=mask, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        image.save(options.output_path)

    def depth_to_image(self, options: DepthToImageOptions) -> None:
        print("---Depth To Image---")
        print(f"Text: {options.prompt}")
        print(f"Number of texts: {options.n_prompt}")
        print(f"Image URL: {options.img_url}")
        print(f"Output Image Path: {options.output_path}")
        prompt = options.prompt
        n_prompt = options.n_prompt
        img_url = options.img_url
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        image = utils.get_image(img_url)
        image = self.pipe(prompt=prompt, image=image, negative_prompt=n_prompt, num_inference_steps=num_inference_steps,
                          width=width, height=height).images[0]
        image.save(options.output_path)
