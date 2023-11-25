import pylla

prompt = "A cute cat"
output_path = "image_cat.png"
num_inference_steps = 5
width = 256
height = 256

pylla.download_model()
pylla.text_to_image(prompt, output_path, num_inference_steps, width, height)
