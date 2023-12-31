import pylla

def main():
    prompt = "Tintin, the adventurous reporter, has uncovered exclusive leaks about Grand Theft Auto VI. In a dimly lit room, surrounded by classified documents and computer screens, Tintin is carefully revealing the details of the highly anticipated game. Rockstar Games' secrets are exposed as Tintin navigates through the virtual world of Grand Theft Auto VI with a sly grin on his face."
    # pylla.generate_tintin_img(prompt, num_inference_steps=20)
    image = pylla.generate_tintin_img(prompt)
    image.save("tintin.png")

if __name__ == "__main__":
    main()
