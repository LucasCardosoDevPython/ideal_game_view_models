import os
import json
import yaml
from PIL import Image
from glob import glob

images_dir = ''
labels_dir = ''
output_json_path = ''

coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 0,
            "name": "board"
        }
    ]
}

