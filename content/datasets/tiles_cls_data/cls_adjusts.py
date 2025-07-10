from albumentations import Illumination, Compose
from random import random, sample
from PIL.Image import open as image_opener
from numpy import array
import matplotlib.pyplot as plt
import os


#clear
for src in ['train', 'val']:
    for folder in os.listdir(src):
        for name in os.listdir(f'{src}/{folder}'):
            os.remove(f'{src}/{folder}/{name}')

#poppulate
data_size = 150
val_threshold = 0.15
src = 'full'

for folder in os.listdir(src):
    if folder == 'BAD':
        names = sample(os.listdir(f'{src}/{folder}'), data_size*3)
    else:
        names = sample(os.listdir(f'{src}/{folder}'), data_size)
    for name in names:
        image = array(image_opener(f'{src}/{folder}/{name}'))
        plt.imsave(
            f'{"val" if random() < val_threshold else "train"}/{folder}/ori_{name}',
            image
        )
        for mode in ["brighten", "darken"]:
            plt.imsave(
                f'{"val" if random()<val_threshold else "train"}/{folder}/{mode[:3]}_{name}',
                Compose([
                    Illumination(
                        mode="linear",
                        intensity_range=[0.01, 0.2],
                        effect_type=mode,
                        angle_range=[0, 360],
                        center_range=[0.1, 0.9],
                        sigma_range=[0.2, 1]
                    )
                ])(image = image)['image']
            )