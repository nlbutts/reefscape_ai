from pathlib import Path
import argparse
import glob
import random
import shutil
import os

def main():
    parser = argparse.ArgumentParser(description='Video2Labels for Yolo')
    parser.add_argument('-o', '--output', help='Output images directory', default='model.trt')
    parser.add_argument('-i', '--input', help='Input directory', required=True)
    parser.add_argument('-p', '--perc', help='Split percent', type=int, default=80)

    args = parser.parse_args()

    os.makedirs(args.output + '/train/images', exist_ok=True)
    os.makedirs(args.output + '/train/labels', exist_ok=True)
    os.makedirs(args.output + '/val/images', exist_ok=True)
    os.makedirs(args.output + '/val/labels', exist_ok=True)

    imgs = glob.glob(args.input + '/images/*')
    for img in imgs:
        p = Path(img)
        base = p.parts[-1]
        stem = p.stem
        label = args.input + '/labels/' + stem + '.txt'

        if random.randint(0, 100) > args.perc:
            outimg = args.output + '/train/images/' + base
            outlabel = args.output + '/train/labels/' + stem + '.txt'
        else:
            outimg = args.output + '/val/images/' + base
            outlabel = args.output + '/val/labels/' + stem + '.txt'

        shutil.copyfile(img, outimg)
        shutil.copyfile(label, outlabel)


if __name__ == "__main__":
    main()