
from pathlib import Path
import numpy as np
from PIL import Image

SIZE = (64,64)
CLASSES = ['square', 'not_square']

def load_folder(folder):
    arrays = []
    for path in sorted(Path(folder).glob('*.png')):
        im = Image.open(path).convert('L').resize(SIZE)
        arr = np.array(im, dtype=np.float32)/ 255.0
        arrays.append(1.0 - arr)
    return arrays

X, y = [], []
for label,  folder in enumerate(CLASSES):
    imgs = load_folder(folder)
    X += imgs
    y += [label] * len(imgs)

X = np.stack(X)[:, None]
y = np.array(y, dtype=np.int64)
np.savez_compressed('dataset.npz', X=X, y=y)
