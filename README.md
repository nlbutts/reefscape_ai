# reefscape_ai

This project is the data files and labels to train a Yolov8n model.

## Steps

1) Use labelme to label new images
2) Run labelme2yolo.py to convert the Labelme labels to YOLO labels
`python3 labelme2yolo.py --category_mapping labels.json images dataset`
3) Setup Ultralytics in a virtualenv:
```
virtualenv -p python3 venv
pip install ultralytics
```
4) Retrain Yolov8n:
`yolo detect train data=/home/nlbutts/projects/reefscape_data/reefscape.yaml model=yolov8n.pt epochs=100 imgsz=640`

5) Test it on a video clip
`yolo detect predict model=runs/detect/train/weights/best.pt source=../reefscape_data/images/PXL_20250107_222204416.TS.mp4`

6) Export to IMX500:
`yolo export format=imx model=runs/detect/train5/weights/best.pt`

This will export directory called **best_imx_model** in the **runs/detect/trainX/weights** directory.

SCP this folder to a Pi and then run TBD to generate a packed model.