import os

from skimage.filters import difference_of_gaussians

from PIL import Image

from numpy import array

import matplotlib.pyplot as plt

from random import random

folders = os.listdir('src')
DIFFERENCE_OF_GAUSSIANS_LOW_SIGMA = 5.5

for folder in folders:
    for name in os.listdir(os.path.join('src', folder)):
        image = Image.open(os.path.join('src', folder, name))
        grayscale = array(image.convert('L')) / 255
        filtered = difference_of_gaussians(grayscale, DIFFERENCE_OF_GAUSSIANS_LOW_SIGMA)

        if random() < -0.1:# change for the frequency u wanna see the results
            fig, ax = plt.subplots(2, figsize=(15, 15))
            ax[0].imshow(image)
            ax[1].imshow(filtered, cmap='gray')
            ax[0].axis('off')
            ax[1].axis('off')
            plt.show()

        plt.imsave(os.path.join('images', folder, name), filtered, cmap='gray')