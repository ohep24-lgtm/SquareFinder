from pathlib import Path
import numpy as np
from PIL import Image

SIZE = (64, 64)
CLASSES = ['square', 'not_square']

class ConvertDir:
    def __init__(self, label, dir, convert=True):
        self.label = label
        self.dir = dir
        self.image_arrays = []
        if convert:
            self.convert()

    def convert(self):
        for path in sorted(Path(self.dir).glob('*.png')):
            im = Image.open(path).convert('L').resize(SIZE)
            arr = np.array(im, dtype=np.float32) / 255.0
            self.image_arrays.append(1.0 - arr)

class CreateDataset:
    def __init__(self, dir_paths):
        self.X, self.y = [], []
        for label, folder in enumerate(dir_paths):
            d = ConvertDir(label, folder)
            self.X += d.image_arrays
            self.y += [label] * len(d.image_arrays)

    def save(self, filename='dataset.npz'):
        X = np.stack(self.X)[:, None]
        y = np.array(self.y, dtype=np.int64)
        np.savez_compressed(filename, X=X, y=y)

if __name__ == '__main__':
    c = CreateDataset(CLASSES)
    c.save()