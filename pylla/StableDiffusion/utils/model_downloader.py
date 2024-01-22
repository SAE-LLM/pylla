import requests
import os
from tqdm import tqdm

model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
output_directory = "pylla/StableDiffusion/models"
output_path = os.path.join(output_directory, "stable-diffusion.ckpt")

class ModelDownloader:
    @staticmethod
    def download_model() -> None:
        # Vérifie si le modèle est déjà téléchargé
        if os.path.exists(output_path):
            print("Model already downloaded")
        else:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            print("Downloading model..")
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk)) # Barre de chargement pour voir l'avancement du téléchargement

            progress_bar.close()
            print("Model downloaded")




if __name__ == "__main__":
    ModelDownloader.download_model()
