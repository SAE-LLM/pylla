from .StableDiffusion.automatic_1111 import call_automatic
from .LLAMA2.llama2 import call_llama

def StableDiffusion():
    return call_automatic()

def LLAMA2():
    return call_llama()