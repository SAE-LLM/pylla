from typing import Dict

class TextToImageOptions:
    def __init__(self, text: str, output_path: str) -> None:
        self.text = text
        self.output_path = output_path

class ImageToTextOptions:
    def __init__(self, image_path: str, output_path: str) -> None:
        self.image_path = image_path
        self.output_path = output_path

class StableDiffusionAI:
    def __init__(self) -> None:
        pass

    def text_to_image(self, options: TextToImageOptions) -> None:
        print("---Text To Image---")
        print(f"Text: {options.text}")
        print(f"Output Image Path: {options.output_path}")

    def image_to_text(self, options: ImageToTextOptions) -> str:
        print("---Image To Text---")
        print(f"Image Path: {options.image_path}")
        print(f"Image Path Result: {options.output_path}")
        return "Text"
