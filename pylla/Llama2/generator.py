from ctransformers import AutoModelForCausalLM


class Llama2AI:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-13B-Ensemble-v5-GGUF",
            model_file="llama-2-13b-ensemble-v5.Q5_K_M.gguf",
            model_type="llama",
            gpu_layers=0
        )

    def generate(self, prompt):
        # URL model LLAMA2: https://huggingface.co/TheBloke/Llama-2-13B-Ensemble-v5-GGUF
        """
        Generates text completion using the LLAMA2 language model.

        Args:
            prompt (str): The input text prompt for text generation.

        Returns:
            str: The generated text completion based on the input prompt.
        """

        response = self.model(prompt)
        return response