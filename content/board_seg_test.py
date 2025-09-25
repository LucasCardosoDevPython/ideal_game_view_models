import os

from cv2 import fillPoly

from PIL import Image
from ultralytics import YOLO
from numpy import array, int32, zeros, uint8

import matplotlib.pyplot as plt

from content import ROOT

model = YOLO(
    os.path.join(ROOT, 'board_seg_runs3/weights/best.pt'),
    task="segment",
    verbose=True
)

testSrc = os.path.join(ROOT, 'datasets/board_seg_data/src/test')

names = os.listdir(testSrc)

def mask_board_contours(model, image):
    """Uses a segmentation model to obtain a binary mask of the board.

    Args:
        model: YOLO segmentation model.
        image: Input image containing the board.

    Returns:
        A binary mask with the board area filled in.
    """
    results = model.predict(
        image,
        verbose=False
    )

    if results[0].masks is None:
        raise RuntimeError('No board found')

    mask = zeros(image.shape[:2], dtype=uint8)

    for segment in results[0].masks.xy:
        points = array(segment).astype(int32)
        fillPoly(mask, [points], 255)

    return mask

for name in names:
    image = Image.open(os.path.join(testSrc, name))
    grayscale = array(image.convert('L').convert('RGB'))
    mask = mask_board_contours(model, grayscale)

    fig, ax = plt.subplots(3, figsize=(15, 15))
    ax[0].imshow(image)
    ax[0].set_title('Original')
    ax[1].imshow(grayscale)
    ax[1].set_title('Grayscale')
    ax[2].imshow(mask)
    ax[2].set_title('Mask')

    plt.show()
