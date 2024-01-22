import pylla

def main():
    prompt = "Code me a function that returns the sum of two numbers"
    print("Verification du modèle en cours...\n")
    response = pylla.llama2_generator(prompt)
    print("Fin de la vérification du modèle.\n")
    print(f"Question -> {prompt}\n")
    print(f"Réponse -> {response}\n")

    print("Verification du modèle tintin en cours...\n")
    prompt = "Tintin, the adventurous reporter, has uncovered exclusive leaks about Grand Theft Auto VI. In a dimly lit room, surrounded by classified documents and computer screens, Tintin is carefully revealing the details of the highly anticipated game. Rockstar Games' secrets are exposed as Tintin navigates through the virtual world of Grand Theft Auto VI with a sly grin on his face."
    image = pylla.generate_tintin_img(prompt)
    image.save("tintin.png")
    
    print("Verification du modèle StableDiffusion en cours...\n")
    prompt = "A cute cat"
    img_url = "pylla/resources/image_bank/forest.jpeg"
    output_path = "image_cat.png"
    num_inference_steps = 200
    #num_inference_steps = 5
    width = 256
    height = 256

    pylla.download_model()
    #pylla.text_to_image(prompt, output_path, num_inference_steps, width, height)
    pylla.image_to_image(prompt, img_url, output_path, num_inference_steps, width, height)


if __name__ == "__main__":
    main()