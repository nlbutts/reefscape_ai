from ultralytics import YOLO
import numpy as np
from pathlib import Path
from PIL import Image
import os
import argparse
import glob

def process(videofile, outputdir, start_num):
    imgpath = Path(outputdir) / 'images'
    labelpath = Path(outputdir) / 'labels'

    os.makedirs(imgpath, exist_ok=True)
    os.makedirs(labelpath, exist_ok=True)

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("reefscape_yolo11x.pt")

    # Perform object detection on an image using the model
    results = model.predict(videofile, show=False)
    for result in results:
        image = Image.fromarray(result.orig_img)
        r, g, b = image.split()
        image = Image.merge("RGB", (b, g, r))
        imgfile = imgpath / f'img_{start_num:06}.bmp'
        labelfile = labelpath / f'img_{start_num:06}.txt'
        image.save(imgfile)

        # Detection
        result = result.to("cpu")
        result.boxes.xyxy  # box with xyxy format, (N, 4)
        result.boxes.xywh  # box with xywh format, (N, 4)
        result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
        result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
        result.boxes.conf  # confidence score, (N, 1)
        result.boxes.cls  # cls, (N, 1)

        # Classification
        result.probs  # cls prob, (num_class, )
        labels = []
        for box in result.boxes:
            if (box.conf > 0.8):
                cls = int(box.cls)
                #bb = np.array(box.xywhn)[0]
                bb = box.xywhn[0]
                s = f'{cls} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\r\n'
                labels.append(s)

        with open(labelfile, 'w') as f:
            f.writelines(labels)

        start_num += 1
    return start_num

def main():
    parser = argparse.ArgumentParser(description='Video2Labels for Yolo')
    parser.add_argument('-o', '--output', help='Output images directory', default='model.trt')
    parser.add_argument('-i', '--input', help='Input directory', required=True)
    parser.add_argument('-n', '--num', help='Starting number', type=int, default=1)

    args = parser.parse_args()

    videos = glob.glob(args.input + '/*')
    count = args.num
    for video in videos:
        print(f'Processing {video}')
        count = process(video, args.output, count)

if __name__ == "__main__":
    main()