import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

def load_polygon_mask_pil(label_txt_path: str, image_size: tuple) -> Image:
    """
    Cria uma máscara binária usando apenas Pillow.
    """
    w, h = image_size
    # Imagem em modo 'L' (preto e branco)
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)

    with open(label_txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * w
            y = coords[i + 1] * h
            points.append((x, y))

        # Desenha o polígono preenchido
        draw.polygon(points, outline=255, fill=255)

    return mask_img

for folder in ['train', 'val']:
    for name in os.listdir(f'images/{folder}'):
        # Caminhos
        image_path = f'images/{folder}/{name}'
        label_path = f'labels/{folder}/{name[:-3]+"txt"}'

        print(f'images/{folder}/{name}')

        # Carrega imagem
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # Gera máscara
        mask_img = load_polygon_mask_pil(label_path, (w, h))

        # Converte para array para visualização
        image_np = np.array(image)
        mask_np = np.array(mask_img)

        # Sobreposição vermelha
        alpha = 0.7
        overlay_np = image_np.copy()
        overlay_np[mask_np > 0] = (255 * alpha + overlay_np[mask_np > 0] * (1 - alpha)).astype(np.uint8)

        # Plota
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_np)
        plt.axis("off")
        plt.title(f"Máscara de Segmentação sobre Imagem {folder}/{name}")
        plt.show()