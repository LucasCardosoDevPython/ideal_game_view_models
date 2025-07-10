from ultralytics import YOLO
import os

from content import ROOT

data_path = 'datasets\\tiles_cls_data'

model = YOLO(os.path.join(data_path, 'yolo11n-cls.pt'))

model.train(
    data= data_path,
    epochs= 30,
    imgsz= 128,
    verbose = True,
    project = ROOT,
    name= 'tiles_cls_runs'
)