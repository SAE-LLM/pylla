from diffusers import StableDiffusionPipeline
import torch

class TextToImageOptions:
    def __init__(self, text: str, output_path: str, num_inference_steps: int, width: int, height: int) -> None:
        self.text = text
        self.output_path = output_path
        self.num_inference_steps  = num_inference_steps 
        self.width  = width 
        self.height  = height 

class StableDiffusionAI:
    def __init__(self, model_id: str = "prompthero/openjourney") -> None:
        self.model_id = model_id
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker = None)

    def text_to_image(self, options: TextToImageOptions) -> None:
        print("---Text To Image---")
        print(f"Text: {options.text}")
        print(f"Output Image Path: {options.output_path}")  
        prompt = options.text
        num_inference_steps = options.num_inference_steps
        width = options.width
        height = options.height
        image = self.pipe(prompt, num_inference_steps=num_inference_steps, width=width, height=height).images[0]
        image.save(options.output_path)