import torch
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline
import pylla.StableDiffusion.utils

@dataclass
class TextToImageOptions:
    """
    Options for text-to-image generation using Stable Diffusion AI.
    """
    prompt: str
    n_prompt: str
    output_path: str
    num_inference_steps: int
    width: int = 256
    height: int = 256

@dataclass
class ImageToImageOptions:
    """
    Options for image-to-image generation using Stable Diffusion AI.
    """
    prompt: str
    n_prompt: str
    img_url: str
    output_path: str
    num_inference_steps: int
    width: int = 256
    height: int = 256

@dataclass
class InPaintingOptions:
    """
    Options for in-painting using Stable Diffusion AI.
    """
    prompt: str
    n_prompt: str
    img_url: str
    mask_url: str
    output_path: str
    num_inference_steps: int
    width: int = 256
    height: int = 256

class StableDiffusionAI:
    """
    Wrapper class for Stable Diffusion AI model.
    """
    def __init__(self, model_path: str = "runwayml/stable-diffusion-v1-5") -> None:
        """
        Initializes the StableDiffusionAI object.

        Args:
            model_path (str): Path to the Stable Diffusion AI model.
        """
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            local_files_only=False
        )

    def text_to_image(self, options: TextToImageOptions) -> None:
        """
        Generates an image from text using Stable Diffusion AI.

        Args:
            options (TextToImageOptions): Options for text-to-image generation.
        """
        print("---Text To Image---")
        print(f"Output Image Path: {options.output_path}")  
        print(f"Text: {options.prompt}")
        prompt = options.prompt
        n_prompt = options.n_prompt
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        if n_prompt:
            image = self.pipe(prompt, num_inference_steps=num_inference_steps, negative_prompt=n_prompt, width=width, height=height).images[0]
        else:
            image = self.pipe(prompt, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        image.save(options.output_path)

    def image_to_image(self, options: ImageToImageOptions) -> None:
        """
        Generates an image from an input image and text using Stable Diffusion AI.

        Args:
            options (ImageToImageOptions): Options for image-to-image generation.
        """
        print("---Image To Image---")
        print(f"Text: {options.prompt}")
        print(f"Image URL: {options.img_url}")
        print(f"Output Image Path: {options.output_path}")
        prompt = self.global_prompt + "\n" + options.prompt        
        n_prompt = options.n_prompt
        img_url = options.img_url
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        image = utils.get_image(img_url)
        if n_prompt:
            image = self.pipe(prompt=prompt, image=image, negative_prompt=n_prompt, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        else:
            image = self.pipe(prompt=prompt, image=image, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        image.save(options.output_path)

    def in_painting(self, options: InPaintingOptions) -> None:
        """
        Performs in-painting using Stable Diffusion AI.

        Args:
            options (InPaintingOptions): Options for in-painting.
        """
        print("---In Painting---")
        print(f"Text: {options.prompt}")
        print(f"Image URL: {options.img_url}")
        print(f"Mask URL: {options.mask_url}")
        print(f"Output Image Path: {options.output_path}")
        prompt = self.global_prompt + "\n" + options.prompt
        n_prompt = options.n_prompt
        img_url = options.img_url
        mask_url = options.mask_url
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        image = utils.get_image(img_url)
        mask = utils.get_image(mask_url)
        if n_prompt:
            image = self.pipe(prompt=prompt, image=image, mask=mask, negative_prompt=n_prompt, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        else:
             image = self.pipe(prompt=prompt, image=image, mask=mask, num_inference_steps=num_inference_steps, width=width, height=height).images[0]

        image.save(options.output_path)
