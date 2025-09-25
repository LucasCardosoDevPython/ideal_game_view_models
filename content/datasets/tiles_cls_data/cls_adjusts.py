import os

from skimage.filters import difference_of_gaussians

from PIL import Image

from numpy import array

import matplotlib.pyplot as plt

from random import random, sample


#clear
for src in ['train', 'val']:
    for folder in os.listdir(src):
        for name in os.listdir(f'{src}/{folder}'):
            os.remove(f'{src}/{folder}/{name}')


folders = os.listdir('full')
DIFFERENCE_OF_GAUSSIANS_LOW_SIGMA = 1
DATASET_SAMPLE_SIZE = 150
VALIDATION_TRESHOLD = 0.15

for folder in folders:
    if folder == 'BAD':
        names = sample(os.listdir(os.path.join('full', folder)), DATASET_SAMPLE_SIZE * 2)
        valNames = sample(names, int(DATASET_SAMPLE_SIZE * 2 * VALIDATION_TRESHOLD))
    else:
        names = sample(os.listdir(os.path.join('full', folder)), DATASET_SAMPLE_SIZE)
        valNames = sample(names, int(DATASET_SAMPLE_SIZE * VALIDATION_TRESHOLD))
    for name in names:
        saveDirectory = os.path.join('val' if name in valNames else 'train', folder)
        if not os.path.exists(saveDirectory):
            os.makedirs(saveDirectory)

        plt.imsave(
            os.path.join(saveDirectory, name),
            difference_of_gaussians(
                array(Image.open(os.path.join('full', folder, name)).convert('L')) / 255,
                DIFFERENCE_OF_GAUSSIANS_LOW_SIGMA
            ),
            cmap='gray'
        )