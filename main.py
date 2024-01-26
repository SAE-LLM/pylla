import tkinter
import tkinter.messagebox
from tkinter import filedialog
import customtkinter
import pylla
import os
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("green")  # Themes: blue (default), dark-blue, green

class ModelLauncherApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Model Launcher")
        self.geometry(f"{1200}x{700}")
        image = Image.open("pylla.png")
        photo = ImageTk.PhotoImage(image)
        self.iconphoto(False, photo)

        self.current_model = -1

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        #MAIN PANEL

        self.param_prompt_label = customtkinter.CTkLabel(self,  text="Prompt:")
        self.param_prompt_entry = customtkinter.CTkTextbox(self, width=500, height=50)

        self.param_image_label = customtkinter.CTkLabel(self,  text="Image Source (facultatif):")
        self.param_image_entry = customtkinter.CTkLabel(self, width=500, height=100, text="Cliquez sur parcourir...")
        self.browse_button = customtkinter.CTkButton(self, text="Parcourir", command=self.browse_image, width=14)
        self.param_num_steps_label = customtkinter.CTkLabel(self,  text="Nombre d'étapes d'inférence:")
        self.param_num_steps_entry = customtkinter.CTkEntry(self, width=300)

        self.param_width_label = customtkinter.CTkLabel(self,  text="Largeur de l'image:")
        self.param_width_entry = customtkinter.CTkEntry(self, width=300)

        self.param_height_label = customtkinter.CTkLabel(self,  text="Hauteur de l'image:")
        self.param_height_entry = customtkinter.CTkEntry(self, width=300)

        self.progress_status = customtkinter.CTkLabel(self, text="En cours...")

        self.launch_button = customtkinter.CTkButton(self, text="Lancer", command=self.launch_selected_model)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="PyLLa", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_llama = customtkinter.CTkButton(self.sidebar_frame, text="Llama2", command=lambda: self.sidebar_button_event(0))
        self.sidebar_button_llama.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_sd = customtkinter.CTkButton(self.sidebar_frame, text="StableDiffusion", command=lambda: self.sidebar_button_event(1))
        self.sidebar_button_sd.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_helsinki = customtkinter.CTkButton(self.sidebar_frame, text="Helsinki", command=lambda: self.sidebar_button_event(2))
        self.sidebar_button_helsinki.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["System", "Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%", "150%", "200%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

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

    def browse_image(self):
        file_path = filedialog.askopenfilename()
        self.param_image_entry.configure(text=file_path)

    def launch_selected_model(self):
        prompt = self.param_prompt_entry.get("0.0", "end")
        print(self.current_model)
        if self.current_model == 0:
            self.launch_llama2_model(prompt)
        elif self.current_model == 1:
            self.launch_stable_diffusion_model(prompt)
        elif self.current_model == 2:
            self.launch_helsinki_model(prompt)

    def launch_llama2_model(self, prompt):
        print("Lancement du modèle Llama2...")
        response = pylla.llama2_generator(prompt)
        print(f"Réponse -> {response}\n")
        tkinter.messagebox.showinfo("LLama2: ", response)
        print("Fin du modèle Llama2.\n")

    def launch_stable_diffusion_model(self, prompt):
        print("Lancement du modèle StableDiffusion...")
        image_source = None
        if self.param_image_entry.cget("text") and self.param_image_entry.cget("text") != "Cliquez sur parcourir..." :
            image_source = self.param_image_entry.cget("text")
        num_inference_steps = int(self.param_num_steps_entry.get())
        width = int(self.param_width_entry.get()) // 8 * 8  # La largeur doit être un multiple de 8
        height = int(self.param_height_entry.get()) // 8 * 8  # La hauteur doit être un multiple de 8

        if image_source:
            pylla.image_to_image(prompt, image_source, "output_image_stableDiffusion.png", num_inference_steps, width, height)
            tkinter.messagebox.showinfo("StableDiffusion", "Image Successfully generated")
        else:
            pylla.text_to_image(prompt, "output_image_stableDiffusion.png", num_inference_steps, width, height)
            tkinter.messagebox.showinfo("StableDiffusion", "Image Successfully generated")


        print("Fin du modèle StableDiffusion.\n")

    def launch_helsinki_model(self, prompt):
            print("Lancement du modèle Helsinki...")
            self.progress_status.grid(row=7, column=2)      
            response = pylla.helsinki_generator(prompt)
            self.progress_status.grid_remove()
            print(f"Réponse -> {response}\n")
            tkinter.messagebox.showinfo("Réponse Helsinki: ", response)
            print("Fin du modèle Helsinki.\n")

    def sidebar_button_event(self, model):
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
        self.progress_status.grid_remove()

        # Afficher les champs nécessaires en fonction du modèle sélectionné
        if model == 0:
            self.current_model = 0
            self.param_prompt_label.grid(row=0, column=1, padx=(5, 0), pady=(2, 0), sticky="w")
            self.param_prompt_entry.grid(row=0, column=2, padx=(10, 10), pady=(2, 0), sticky="nsew")
            self.param_num_steps_label.grid(row=1, column=1, pady=5, sticky="w")
            self.param_num_steps_entry.grid(row=1, column=2, pady=5, sticky="e")
        elif model == 1:
            self.current_model = 1
            self.param_prompt_label.grid(row=0, column=1, padx=(5, 0), pady=(2, 0), sticky="w")
            self.param_prompt_entry.grid(row=0, column=2, padx=(10, 10), pady=(2, 0), sticky="nsew")
            self.param_image_label.grid(row=1, column=1, pady=5, sticky="w")
            self.param_image_entry.grid(row=1, column=2, pady=10, sticky="nsew")
            self.browse_button.grid(row=1, column=3, pady=5, sticky="e")
            self.param_num_steps_label.grid(row=2, column=1, pady=5, sticky="w")
            self.param_num_steps_entry.grid(row=2, column=2, pady=5, sticky="e")
            self.param_width_label.grid(row=3, column=1, pady=5, sticky="w")
            self.param_width_entry.grid(row=3, column=2, pady=5, sticky="nsew")
            self.param_height_label.grid(row=4, column=1, pady=5, sticky="w")
            self.param_height_entry.grid(row=4, column=2, pady=5, sticky="nsew")
        elif model == 2:
            self.current_model = 2
            self.param_prompt_label.grid(row=0, column=1, padx=(5, 0), pady=(2, 0), sticky="w")
            self.param_prompt_entry.grid(row=0, column=2, padx=(10, 10), pady=(2, 0), sticky="nsew")

        self.launch_button.grid(row=6, column=2)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)


def main():
    app = ModelLauncherApp()
    app.mainloop()

if __name__ == "__main__":
    main()
