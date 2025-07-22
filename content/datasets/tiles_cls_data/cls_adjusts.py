from albumentations import Illumination, Compose, RandomBrightnessContrast
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
            f'{"val" if random() < val_threshold else "train"}/{folder}/0_{name}',
            image
        )
        for i in range(5):
            plt.imsave(
                f'{"val" if random()<val_threshold else "train"}/{folder}/{i+1}_{name}',
                Compose([
                    RandomBrightnessContrast(
                        brightness_limit=[-0.1, -0.5],
                        contrast_limit=[0, 0],
                        brightness_by_max=False,
                        ensure_safe_range=False
                    ),
                    Illumination(
                        mode="gaussian",
                        intensity_range=[0.1, 0.2],
                        effect_type="darken",
                        angle_range=[0, 360],
                        center_range=[0.1, 0.9],
                        sigma_range=[0.2, 1]
                    ),
                ])(image = image)['image']
            )