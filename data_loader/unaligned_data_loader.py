import random
import torch.utils.data
import torchvision.transforms as transforms
# from base_data_loader import BaseDataLoader
# import torchnet as tnt
# pip install future --upgrade
from builtins import object
from pdb import set_trace as st
import torch.utils.data as data_utils


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths, A_idx = None, None, None
        B, B_paths, B_idx = None, None, None
        try:
            A, A_paths, A_idx = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None or A_idx is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths, A_idx = next(self.data_loader_A_iter)

        try:
            B, B_paths, B_idx = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None or B_idx is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths, B_idx = next(self.data_loader_B_iter)
        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': A, 'S_label': A_paths, 'S_idx':A_idx,
                    'T': B, 'T_label': B_paths, 'T_idx':B_idx}


class PairedData_3(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_C, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_C = data_loader_C
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.data_loader_C_iter = iter(self.data_loader_C)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths, A_idx = None, None, None
        B, B_paths, B_idx = None, None, None
        C, C_paths, C_idx = None, None, None
        try:
            A, A_paths, A_idx = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None or A_idx is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths, A_idx = next(self.data_loader_A_iter)

        try:
            B, B_paths, B_idx = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None or B_idx is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths, B_idx = next(self.data_loader_B_iter)

        try:
            C, C_paths, C_idx = next(self.data_loader_C_iter)
        except StopIteration:
            if C is None or C_paths is None or C_idx is None:
                self.stop_C = True
                self.data_loader_C_iter = iter(self.data_loader_C)
                C, C_paths, C_idx = next(self.data_loader_C_iter)

        if (self.stop_A and self.stop_B and self.stop_C) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            self.stop_C = False
            raise StopIteration()
        else:
            self.iter += 1
            img = torch.cat((A, B))
            label = torch.cat((A_paths, B_paths))
            idx = torch.cat((A_idx, B_idx))
            return {'S': img, 'S_label': label, 'S_idx':idx,
                    'T': C, 'T_label': C_paths, 'T_idx':C_idx}


class UnalignedDataLoader_3():
    def initialize(self, A, B, C, batchSize):
        # BaseDataLoader.initialize(self)
        dataset_A = A  # tnt.dataset.TensorDataset([A['features'], A['targets']])
        dataset_B = B  # tnt.dataset.TensorDataset([B['features'], B['targets']])
        dataset_C = C
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=int(batchSize / 2),
            shuffle=True,
            # not self.serial_batches,
            num_workers=4)

        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=int(batchSize / 2),
            shuffle=True,
            # not self.serial_batches,
            num_workers=4)

        data_loader_C = torch.utils.data.DataLoader(
            dataset_C,
            batch_size=batchSize,
            shuffle=True,
            # not self.serial_batches,
            num_workers=4)

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.dataset_C = dataset_C
        flip = False  # opt.isTrain and not opt.no_flip
        self.paired_data = PairedData_3(data_loader_A, data_loader_B, data_loader_C, float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A) * 2, len(self.dataset_B) * 2, len(self.dataset_C)),
                   self.paired_data.max_dataset_size)


class UnalignedDataLoader():
    def initialize(self, A, B, batchSize):
        # BaseDataLoader.initialize(self)
        dataset_A = A  # tnt.dataset.TensorDataset([A['features'], A['targets']])
        dataset_B = B  # tnt.dataset.TensorDataset([B['features'], B['targets']])
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=batchSize,
            shuffle=True,
            # not self.serial_batches,
            num_workers=4, drop_last=True)

        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=batchSize,
            shuffle=True,
            # not self.serial_batches,
            num_workers=4, drop_last=True)

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        flip = False  # opt.isTrain and not opt.no_flip
        self.paired_data = PairedData(data_loader_A, data_loader_B, float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_A), len(self.dataset_B)), self.paired_data.max_dataset_size)
