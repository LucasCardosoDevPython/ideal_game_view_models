from ultralytics import YOLO
import os

from content import ROOT

data_path = 'datasets\\board_seg_data'

model = YOLO(os.path.join(data_path, 'yolo11n-seg.pt'))

model.train(
    data= f'{data_path}\\data.yaml',
    epochs= 30,
    verbose = True,
    project = ROOT,
    name= 'board_seg_runs'
)