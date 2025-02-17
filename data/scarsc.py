import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances

from config import scars_root, scarsc_root, scars_meta_path, severity, distortions


class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=scars_root, transform=None, metas=scars_meta_path):

        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


class CarscDataset(Dataset):
    
    def __init__(self, data_dir, ori_dir, base_folder=None, train=True, limit=0, transform=None, metas=scars_meta_path):

        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.ori_dir = ori_dir
        self.base_folder = base_folder

        self.data_idx = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data_idx.append(img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx]
        path = os.path.join(self.data_dir, self.base_folder, data_idx)

        image = self.loader(path)
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data_idx)


class MergedCarscDataset(Dataset):
    
    def __init__(self, data_dir, ori_dir, args=None, train=True, limit=0, transform=None, metas=scars_meta_path):

        data_dir = data_dir.format('train') if train else data_dir.format('test')
        metas = metas.format('train_annos') if train else metas.format('test_annos_withlabels')

        self.loader = default_loader
        self.data_dir = data_dir
        self.ori_dir = ori_dir
        self.args = args

        self.data_idx = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data_idx.append(img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):
        sev = str(idx // (len(self.data_idx) * len(distortions))+1)
        dist = distortions[idx // len(self.data_idx) % len(distortions)]
        data_idx = self.data_idx[idx-((int(sev)-1)*len(self.data_idx)*len(distortions)+(idx//len(self.data_idx))*len(self.data_idx))]

        path = os.path.join(self.data_dir, dist, sev, data_idx)

        image = self.loader(path)
        target = self.target[idx%len(self.data_idx)] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data_idx) * len(distortions) * len(severity)
    

def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_scarsc_datasets(train_transform, test_transform, train_classes=range(160), prop_train_labels=0.8,
                    split_train_val=False, seed=0, args=None):

    np.random.seed(seed)

    if args.only_test:
        test_dataset = CarscDataset(data_dir=scarsc_root, ori_dir=scars_root, base_folder=args.distortion+'/'+args.severity, transform=test_transform, metas=scars_meta_path, train=False)
        all_datasets = {
            'test': test_dataset,
        }
        return all_datasets

    if args.task_type == 'A_L+A_U->B':
        # Init entire training set
        whole_training_set = CarsDataset(data_dir=scars_root, transform=train_transform, metas=scars_meta_path, train=True)

        # Get labelled training set which has subsampled classes, then subsample some indices from that
        train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
        subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

        # Split into training and validation sets
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

        # Get unlabelled data
        unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

        # Either split train into train and val or use test set as val
        train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'trainA_unlabelled': train_dataset_unlabelled,
            'val': val_dataset_labelled,
        }

    elif args.task_type == 'A_L+A_U+B->A_U+B+C':
        whole_training_set = CarsDataset(data_dir=scars_root, transform=train_transform, metas=scars_meta_path, train=True)
        train_datasetB_unlabelled = MergedCarscDataset(data_dir=scarsc_root, ori_dir=scars_root, args=args, transform=train_transform, metas=scars_meta_path, train=False)
        test_datasetB = MergedCarscDataset(data_dir=scarsc_root, ori_dir=scars_root, args=args, transform=test_transform, metas=scars_meta_path, train=False)

        # Get labelled training set which has subsampled classes, then subsample some indices from that
        train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
        subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
        train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

        # Split into training and validation sets
        train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
        train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
        val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
        val_dataset_labelled_split.transform = test_transform

        # Get unlabelled data
        unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
        train_datasetA_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

        # Either split train into train and val or use test set as val
        train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
        val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

        all_datasets = {
            'train_labelled': train_dataset_labelled,
            'trainA_unlabelled': train_datasetA_unlabelled,
            'trainB_unlabelled': train_datasetB_unlabelled,
            'val': val_dataset_labelled,
            'testB': test_datasetB,
        }

    return all_datasets


if __name__ == '__main__':

    x = get_scarsc_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')