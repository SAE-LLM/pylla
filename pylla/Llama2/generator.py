from ctransformers import AutoModelForCausalLM

def llama2_generator(prompt):
    # URL model LLAMA2: https://huggingface.co/TheBloke/Llama-2-13B-Ensemble-v5-GGUF
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-13B-Ensemble-v5-GGUF",
        model_file="llama-2-13b-ensemble-v5.Q5_K_M.gguf",
        model_type="llama",
        gpu_layers=0
    )

    response = model(prompt)
    return response