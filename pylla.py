import pylla
import os

class Pylla:
    def __init__(self):
        # MODELS
        self.__llama2 = pylla.Llama2AI()
        self.__stable_diffusion = pylla.StableDiffusionAI()
        self.__helsinki = pylla.HelsinkiAI()

        # PARAMS

            # PROMPTS
        self.__global_prompt = None
        self.__style_prompt = None

            # OTHER
        self.__inference = 20

    def reset_llama2():
        self.__llama2 = pylla.Llama2AI()

    def reset_stable_diffusion():
        self.__stable_diffusion = pylla.StableDiffusionAI()

    def reset_helsinki():
        self.__helsinki = pylla.HelsinkiAI()

    def reset_all():
        reset_llama2()
        reset_stable_diffusion()
        reset_helsinki()

    def set_header_prompts(self, positive_prompt=None, global_prompt=None,  negative_prompt=None, style_prompt=None):
        if global_prompt:
            self.__global_prompt = global_prompt
        if style_prompt:
            self.style_prompt = style_prompt

    def set_inference(self, inference=None, width=None, height=None):
        if inference:
            self.__inference = inference

    def llama2(self, positive_prompt, negative_prompt=None):
        prompt = str(self.__global_prompt) + " " +  str(self.style_prompt) + " " + str(positive_prompt) + " " + str(negative_prompt)

        self.__display_infos(0, prompt)
        res = self.__llama2.generate(prompt)

        return res
        

    def stable_diffusion_txt2img(self, positive_prompt, negative_prompt=None, output_path="output_txt2img.png", width=256, height=256):
        prompt = str(self.__global_prompt) + " " +  str(self.style_prompt)
        n_prompt = str(negative_prompt)
        if not output_path.endswith(".png"):
            output_path += ".png"

        options = pylla.TextToImageOptions(
            prompt=prompt,
            n_prompt=negative_prompt,
            output_path=output_path,
            num_inference_steps=self.__inference,
            width=width,
            height=height
        )

        self.__display_infos(1, prompt, n_prompt)
        self.__stable_diffusion.text_to_image(options)

    def stable_diffusion_img2img(self, positive_prompt, img_path, negative_prompt=None, output_path="output_txt2img.png", width=256, height=256):
        prompt = str(self.__global_prompt) + " " +  str(self.style_prompt)
        n_prompt = str(negative_prompt)
        if not output_path.endswith(".png"):
            output_path += ".png"

        options = pylla.ImageToImageOptions(
            prompt=prompt,
            n_prompt=negative_prompt,
            img_url=img_path,
            output_path=output_path,
            num_inference_steps=self.__inference,
            width=width,
            height=height
        )

        self.__display_infos(1, prompt, n_prompt)
        self.__stable_diffusion.image_to_image(options)

    def stable_diffusion_inpainting(self, positive_prompt, img_path, mask_path, negative_prompt=None, output_path="output_txt2img.png", width=256, height=256):
        prompt = str(self.__global_prompt) + " " +  str(self.style_prompt)
        n_prompt = str(negative_prompt)
        if not output_path.endswith(".png"):
            output_path += ".png"

        options = pylla.InPaintingOptions(
            prompt=prompt,
            n_prompt=negative_prompt,
            img_url=img_path,
            mask_url=mask_path,
            output_path=output_path,
            num_inference_steps=self.__inference,
            width=width,
            height=height
        )
        
        self.__display_infos(1, prompt, n_prompt)
        self.__stable_diffusion.in_painting(options)

    def helsinki(self, prompt):
        
        self.__display_infos(2, prompt)
        res = self.__helsinki.generate(prompt)
        return res

    def __display_infos(self, model, p_prompt, n_prompt=None, width=None, height=None):
        print("==== Pylla ====")
        if model == 0:
            print("\nUsing Llama2:")
        elif model == 1:
            print("\nUsing StableDiffusion:")
        elif model == 2:
            print("\nUsing Helinski:")
        print("\nParams:")
        print(f" - Global Prompt: {self.__global_prompt}")
        print(f" - Style Prompt: {self.__style_prompt}")
        print(f" - Positive Prompt: {p_prompt}")

        
        if model == 1:
            print(f" - Negative Prompt: {n_prompt}")
            print(f" - Inference Steps: {self.__inference}")
            print(f" - Width: {width}")
            print(f" - Height: {height}")



        

def main():
    """
    Main function to run the Model Launcher application.
    """

    pylla = Pylla()

    #TRADUCTION
    prompt = "J'utilise Helsinki pour traduire mon texte!"
    
    res = pylla.helsinki(prompt)

    print(f"Result: {res}")




if __name__ == "__main__":
    main()
