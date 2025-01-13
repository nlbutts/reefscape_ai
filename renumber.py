import argparse
import glob
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description='Onnx perf')
parser.add_argument('-s', '--start', help='Start number', type=int, required=True)
parser.add_argument('-i', '--input', help='Input directory', type=str, required=True)
parser.add_argument('-o', '--output', help='Output directory', type=str, required=True)
args = parser.parse_args()

files = glob.glob(args.input + '/*.bmp')
print(f'Found files {files}')

outdir = Path(args.output)
outdir.mkdir(exist_ok=True)

startnum = args.start
for file in files:
    outfile = outdir / f'img_{startnum:06}.bmp'
    shutil.copy(file, str(outfile))
    startnum += 1
