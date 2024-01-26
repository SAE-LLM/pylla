from PIL import Image
import requests
import os
from io import BytesIO

"""
    Récupère une image depuis une URL ou un chemin local
    :param url: URL ou chemin local de l'image
    :return: Image
"""
def get_image(url: str) -> Image:
    if os.path.isfile(url):
        image = Image.open(url)
    else:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    return image
