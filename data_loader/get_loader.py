from .mydataset import ImageFolder, ImageFilelist
from .unaligned_data_loader import UnalignedDataLoader, UnalignedDataLoader_3
import os
import torch
from torch.utils.data import DataLoader


def get_loader_test(source_path, target_path, evaluation_path, transforms, batch_size=32):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False,train=True)
    eval_folder_test = ImageFilelist(os.path.join(evaluation_path),
                                     '/data/ugui0/ksaito/VISDA_tmp/image_list_val.txt',
                                     transform=transforms[evaluation_path],
                                     return_paths=True)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, test_loader


def get_loader_2(source_path1, source_path2, target_path, root_path, evaluation_path, transforms, batch_size=32):
    source_folder1 = ImageFolder(os.path.join(source_path1), root_path,
                                transforms[source_path1], return_paths=True)
    source_folder2 = ImageFolder(os.path.join(source_path2), root_path,
                                 transforms[source_path2], return_paths=True)
    target_folder_train = ImageFolder(os.path.join(target_path), root_path,
                                      transform=transforms[target_path],
                                      return_paths=True)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path), root_path,
                                     transform=transforms[evaluation_path],
                                     return_paths=True)

    train_loader1 = UnalignedDataLoader_3()
    train_loader1.initialize(source_folder1, source_folder2, target_folder_train, batch_size)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8)

    return train_loader1, test_loader


def get_pseudo_loader(pseudo_file, args, transform, p):
    pseudo_folder = ImageFolder(pseudo_file, args.root_path, transform, return_paths=True)
    pseudo_loader = torch.utils.data.DataLoader(pseudo_folder, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                                drop_last=True)

    return pseudo_loader

def get_loader(source_path, target_path, root_path, evaluation_path, transforms, batch_size=32):
    source_folder = ImageFolder(os.path.join(source_path), root_path,
                                transforms[source_path], return_paths=True)
    target_folder_train = ImageFolder(os.path.join(target_path), root_path,
                                      transform=transforms[target_path],
                                      return_paths=True)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path), root_path,
                                     transform=transforms[evaluation_path],
                                     return_paths=True)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, test_loader
