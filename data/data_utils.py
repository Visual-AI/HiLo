import numpy as np
from numpy import cumsum
from torch.utils.data import Dataset

from bisect import bisect


def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices


def subsample_balanced_instances(dataset, prop_indices_to_subsample=0.5):
    np.random.seed(0)

    targets_dict = {}  # idx of each label

    for idx, l in enumerate(dataset.targets):
        if l in targets_dict.keys():
            if not idx in targets_dict[l]:
                targets_dict[l].append(idx)
        else:
            targets_dict[l] = [idx]

    #divide
    picked_indices = np.array([])

    for key in targets_dict:
        subsample_indices = np.random.choice(range(len(targets_dict[key])), replace=False,
                                            size=(int(prop_indices_to_subsample * len(targets_dict[key])),))
        mask = np.zeros(len(targets_dict[key])).astype('bool')
        mask[subsample_indices] = True
        picked_indices = np.append(picked_indices, np.array(targets_dict[key])[mask])

    return picked_indices.astype(int)


class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


class MergedTriDataset(Dataset):

    """
    Takes three datasets (labelled_dataset, unlabelled_dataset1, unlabelled_dataset2) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset1, unlabelled_dataset2):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset1 = unlabelled_dataset1
        self.unlabelled_dataset2 = unlabelled_dataset2
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        elif item >= len(self.labelled_dataset) and item < len(self.labelled_dataset) + len(self.unlabelled_dataset1):
            img, label, uq_idx = self.unlabelled_dataset1[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        else:
            img, label, uq_idx = self.unlabelled_dataset2[item - len(self.labelled_dataset) - len(self.unlabelled_dataset1)]
            labeled_or_not = 0

        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset1) + len(self.unlabelled_dataset2) + len(self.labelled_dataset)