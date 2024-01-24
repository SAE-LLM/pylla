import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pylla

class ModelLauncherApp:
    def __init__(self, master):
        self.master = master
        master.title("Model Launcher")
        master.geometry("600x300")

        self.model_selector_label = ttk.Label(master, text="Choisissez un modèle:")
        self.model_selector = ttk.Combobox(master, values=["Tintin", "Llama2", "StableDiffusion"])

        self.param_prompt_label = ttk.Label(master, text="Prompt:")
        self.param_prompt_entry = ttk.Entry(master, width=50)

        self.param_image_label = ttk.Label(master, text="Image Source (facultatif):")
        self.param_image_entry = ttk.Entry(master, width=34)
        self.browse_button = ttk.Button(master, text="Parcourir", command=self.browse_image, width=14)

        self.param_num_steps_label = ttk.Label(master, text="Nombre d'étapes d'inférence:")
        self.param_num_steps_entry = ttk.Entry(master, width=50)

        self.param_width_label = ttk.Label(master, text="Largeur de l'image:")
        self.param_width_entry = ttk.Entry(master, width=50)

        self.param_height_label = ttk.Label(master, text="Hauteur de l'image:")
        self.param_height_entry = ttk.Entry(master, width=50)

        self.launch_button = ttk.Button(master, text="Lancer", command=self.launch_selected_model)

        # Gestion de l'affichage dynamique
        self.model_selector_label.grid(row=0, column=0, pady=10, sticky="w")
        self.model_selector.grid(row=0, column=1, pady=10)

        # Masquer les champs initialement
        self.param_prompt_label.grid_remove()
        self.param_prompt_entry.grid_remove()
        self.param_image_label.grid_remove()
        self.param_image_entry.grid_remove()
        self.browse_button.grid_remove()
        self.param_num_steps_label.grid_remove()
        self.param_num_steps_entry.grid_remove()
        self.param_width_label.grid_remove()
        self.param_width_entry.grid_remove()
        self.param_height_label.grid_remove()
        self.param_height_entry.grid_remove()
        self.launch_button.grid_remove()

        # Gestion de l'affichage dynamique
        self.model_selector.bind("<<ComboboxSelected>>", self.update_interface)

    def browse_image(self):
        file_path = filedialog.askopenfilename()
        self.param_image_entry.delete(0, tk.END)
        self.param_image_entry.insert(0, file_path)

    def update_interface(self, event):
        selected_model = self.model_selector.get()

        # Masquer tous les champs
        self.param_prompt_label.grid_remove()
        self.param_prompt_entry.grid_remove()
        self.param_image_label.grid_remove()
        self.param_image_entry.grid_remove()
        self.browse_button.grid_remove()
        self.param_num_steps_label.grid_remove()
        self.param_num_steps_entry.grid_remove()
        self.param_width_label.grid_remove()
        self.param_width_entry.grid_remove()
        self.param_height_label.grid_remove()
        self.param_height_entry.grid_remove()
        self.launch_button.grid_remove()

        # Afficher les champs nécessaires en fonction du modèle sélectionné
        if selected_model == "Tintin":
            self.param_prompt_label.grid(row=1, column=0, pady=5, sticky="w")
            self.param_prompt_entry.grid(row=1, column=1, pady=5, sticky="w")
        elif selected_model == "Llama2":
            self.param_prompt_label.grid(row=1, column=0, pady=5, sticky="w")
            self.param_prompt_entry.grid(row=1, column=1, pady=5, sticky="w")
            self.param_num_steps_label.grid(row=2, column=0, pady=5, sticky="w")
            self.param_num_steps_entry.grid(row=2, column=1, pady=5, sticky="w")
        elif selected_model == "StableDiffusion":
            self.param_prompt_label.grid(row=1, column=0, pady=5, sticky="w")
            self.param_prompt_entry.grid(row=1, column=1, pady=5, sticky="w")
            self.param_image_label.grid(row=2, column=0, pady=5, sticky="w")
            self.param_image_entry.grid(row=2, column=1, pady=5, sticky="w")
            self.browse_button.grid(row=2, column=1, pady=5, sticky="e")
            self.param_num_steps_label.grid(row=3, column=0, pady=5, sticky="w")
            self.param_num_steps_entry.grid(row=3, column=1, pady=5, sticky="w")
            self.param_width_label.grid(row=4, column=0, pady=5, sticky="w")
            self.param_width_entry.grid(row=4, column=1, pady=5, sticky="w")
            self.param_height_label.grid(row=5, column=0, pady=5, sticky="w")
            self.param_height_entry.grid(row=5, column=1, pady=5, sticky="w")

        self.launch_button.grid(row=6)

    def launch_selected_model(self):
        selected_model = self.model_selector.get()
        prompt = self.param_prompt_entry.get()

        if selected_model == "Tintin":
            self.launch_tintin_model(prompt)
        elif selected_model == "Llama2":
            self.launch_llama2_model(prompt)
        elif selected_model == "StableDiffusion":
            self.launch_stable_diffusion_model(prompt)

    def launch_tintin_model(self, prompt):
        print("Lancement du modèle Tintin...")
        num_inference_steps = int(self.param_num_steps_entry.get())
        image = pylla.generate_tintin_img(prompt, num_inference_steps)
        image.save("output_image_tintin.png")
        print("Fin du modèle Tintin.\n")

    def launch_llama2_model(self, prompt):
        print("Lancement du modèle Llama2...")
        response = pylla.llama2_generator(prompt)
        print(f"Réponse -> {response}\n")
        print("Fin du modèle Llama2.\n")

    def launch_stable_diffusion_model(self, prompt):
        print("Lancement du modèle StableDiffusion...")
        image_source = self.param_image_entry.get()
        num_inference_steps = int(self.param_num_steps_entry.get())
        width = int(self.param_width_entry.get()) // 8 * 8  # La largeur doit être un multiple de 8
        height = int(self.param_height_entry.get()) // 8 * 8  # La hauteur doit être un multiple de 8

        if image_source:
            pylla.image_to_image(prompt, image_source, "output_image_stableDiffusion.png", num_inference_steps, width, height)
        else:
            pylla.text_to_image(prompt, "output_image_stableDiffusion.png", num_inference_steps, width, height)

        print("Fin du modèle StableDiffusion.\n")


def main():
    root = tk.Tk()
    app = ModelLauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
