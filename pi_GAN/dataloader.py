import torch
import numpy as np
import os
import random
import imageio
from PIL import Image
from tqdm import tqdm

class DataLoader:
    def __init__(self, data_path, batch_size, resize=1.0, preload=False, data_num=None):
        self.need_preload = preload
        self.resize = resize
        self.data_path = data_path
        self.data_files = [os.path.join(data_path, f) for f in sorted(os.listdir(data_path)) if 'png' in f]
        if data_num is not None:
            self.data_files = self.data_files[:data_num]
        self.dataset = None
        if self.need_preload:
            self.preload()
        self.shuffle()
        self.n_data_files = len(self.data_files)
        self.epoch = 0
        self.batch = 0
        self.batch_size = batch_size

    def preload(self):
        print('[INFO] DataLoader: reading data files...')
        self.dataset = []
        for file_name in tqdm(self.data_files):
            image = Image.open(file_name)
            if self.resize != 1:
                image = image.resize((int(self.resize * image.width), int(self.resize * image.height)), Image.ANTIALIAS)
            self.dataset.append(np.array(image, dtype=np.float32))
        self.dataset = np.stack(self.dataset, axis=0).astype(np.float32) / 255
        self.dataset = torch.tensor(self.dataset, dtype=torch.float, device='cuda')

    def shuffle(self):
        if self.need_preload:
            shuffle_idx = torch.randperm(self.dataset.shape[0])
            self.dataset = self.dataset[shuffle_idx]
        else:
            random.shuffle(self.data_files)

    def set_batch_size(self, batch_size):
        self.shuffle()
        self.epoch = 0
        self.batch = 0
        self.batch_size = batch_size

    def get(self):
        epoch, batch = self.epoch, self.batch
        start = self.batch * self.batch_size
        end = min((self.batch + 1) * self.batch_size, self.n_data_files)
        if self.need_preload:
            batch_data = self.dataset[start:end]
        else:
            batch_files = self.data_files[start:end]
            batch_data = []
            for file_name in batch_files:
                image = Image.open(file_name)
                if self.resize != 1:
                    image = image.resize((int(self.resize * image.width), int(self.resize * image.height)), Image.ANTIALIAS)
                batch_data.append(np.array(image, dtype=np.float32))
            batch_data = np.stack(batch_data, axis=0).astype(np.float32) / 255
            batch_data = torch.tensor(batch_data, dtype=torch.float, device='cuda')
        self.batch += 1
        if end == self.n_data_files:
            self.shuffle()
            self.epoch += 1
            self.batch = 0
        return epoch, batch, batch_data


if __name__ == '__main__':
    dataset = DataLoader('./data/image64_rescale', batch_size=128, resize=0.5, preload=True)
    for i in range(10000):
        epoch, batch, img = dataset.get()
        print(epoch, batch, img.shape)
