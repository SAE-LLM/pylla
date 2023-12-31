import pylla

def main():
    prompt = "Code me a function that returns the sum of two numbers"
    print("Verification du modèle en cours...\n")
    response = pylla.llama2_generator(prompt)
    print("Fin de la vérification du modèle.\n")
    print(f"Question -> {prompt}\n")
    print(f"Réponse -> {response}\n")

if __name__ == "__main__":
    main()
