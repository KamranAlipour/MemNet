import torch.utils.data as data
import torch
import h5py

import glob
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
        
    def __len__(self):
        return self.data.shape[0]


class DatasetFromNpys(data.Dataset):
      def __init__(self, file_path):
         super(DatasetFromNpys, self).__init__()
         self.data = sorted(glob.glob(file_path+'/input*'))
         self.target = sorted(glob.glob(file_path+'/target*'))

      def __getitem__(self, index):
         return torch.from_numpy(np.load(self.data[index])).float(), torch.from_numpy(np.load(self.target[index])).float()

      def __len__(self):
         return len(self.data)
