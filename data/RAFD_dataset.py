from torch.utils.data import Dataset
import data.util_2D as Util
import os
import numpy as np
from skimage import io
import cv2

# class RAFDDataset(Dataset):
#     def __init__(self, dataroot, split='test'):
#         self.split = split
#         self.imageNum = []

#         self.datapath = os.path.join(dataroot, split)
#         dataFiles = sorted(os.listdir(self.datapath))


#         self.imageNum.append([dataFiles[0], dataFiles[1]])
#         self.data_len = len(self.imageNum)

#     def __len__(self):
#         return self.data_len

#     def __getitem__(self, index):
#         fileInfo = self.imageNum[index]
#         dataX, dataY = fileInfo[0], fileInfo[1]
#         dataXPath = os.path.join(self.datapath, dataX)
#         dataYPath = os.path.join(self.datapath, dataY)
#         data = io.imread(dataXPath, as_gray=True).astype(float)[:, :, np.newaxis]
#         label = io.imread(dataYPath, as_gray=True).astype(float)[:, :, np.newaxis]

#         dataX_RGB = io.imread(dataXPath).astype(float)
#         dataY_RGB = io.imread(dataYPath).astype(float)

#         [data, label] = Util.transform_augment([data, label], split=self.split, min_max=(-1, 1))

#         return {'M': data, 'F': label, 'MC': dataX_RGB, 'FC': dataY_RGB, 'nS': 7, 'P':fileInfo, 'Index': index}

import glob


class RAFDDataset(Dataset):
    def __init__(self, dataroot, split='test'):
        self.split = split
        self.imageNum = []

        self.moving_paths = glob.glob(os.path.join(dataroot, split, 'moving', '*'))
        self.fixed_paths = glob.glob(os.path.join(dataroot, split, 'fixed', '*'))

        self.imageNum = []
        for moving_path in self.moving_paths:
            fixed_path = np.random.choice(self.fixed_paths)
            self.imageNum.append((moving_path, fixed_path))
        # self.datapath = os.path.join(dataroot, split)
        # dataFiles = sorted(os.listdir(self.datapath))

        # self.imageNum.append([dataFiles[0], dataFiles[1]])
        # self.imageNum = dataFiles

        self.data_len = len(self.imageNum)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        filePaths = self.imageNum[index]
        fileInfo = os.path.basename(filePaths[0]), os.path.basename(filePaths[1])
        dataXPath, dataYPath = filePaths[0], filePaths[1]
        # dataXPath = os.path.join(self.datapath, dataX)
        # dataYPath = os.path.join(self.datapath, dataY)

        size = 128
        data = cv2.imread(dataXPath, 0)
        data = cv2.resize(data, (size, size))
        data = data.astype(float)[:, :, np.newaxis]

        label = cv2.imread(dataYPath, 0)
        label = cv2.resize(label, (size, size))
        label = label.astype(float)[:, :, np.newaxis]

        dataX_RGB = cv2.imread(dataXPath, 1)[:, :, ::-1]
        dataX_RGB = cv2.resize(dataX_RGB, (size, size))
        dataX_RGB = dataX_RGB.astype(float)

        dataY_RGB = cv2.imread(dataYPath, 1)[:, :, ::-1]
        dataY_RGB = cv2.resize(dataY_RGB, (size, size))
        dataY_RGB = dataY_RGB.astype(float)

        # data = io.imread(dataXPath, as_gray=True).astype(float)[:, :, np.newaxis]
        # label = io.imread(dataYPath, as_gray=True).astype(float)[:, :, np.newaxis]

        # dataX_RGB = io.imread(dataXPath).astype(float)
        # dataY_RGB = io.imread(dataYPath).astype(float)

        [data, label] = Util.transform_augment([data, label], split=self.split, min_max=(-1, 1))

        return {'M': data, 'F': label, 'MC': dataX_RGB, 'FC': dataY_RGB, 'nS': 7, 'P':fileInfo, 'Index': index}
