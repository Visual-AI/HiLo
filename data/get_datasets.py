from data.data_utils import MergedDataset, MergedTriDataset
from data.domainnet import get_domainnet_datasets
from data.cubc import get_cubc_datasets
from data.scarsc import get_scarsc_datasets
from data.fgvcc import get_fgvcc_datasets

from data.domainnet import subsample_classes as subsample_dataset_domainnet
from data.cubc import subsample_classes as subsample_dataset_cubc
from data.scarsc import subsample_classes as subsample_dataset_scarsc
from data.fgvcc import subsample_classes as subsample_dataset_fgvcc

from copy import deepcopy
import pickle
import os

from config import osr_split_dir

sub_sample_class_funcs = {
    'domainnet': subsample_dataset_domainnet,
    'cubc': subsample_dataset_cubc,
    'fgvcc': subsample_dataset_fgvcc,
    'scarsc': subsample_dataset_scarsc
}

get_dataset_funcs = {
    'domainnet': get_domainnet_datasets,
    'cubc': get_cubc_datasets,
    'fgvcc': get_fgvcc_datasets,
    'scarsc': get_scarsc_datasets
}


def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            args=args)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    if args.only_test:
        return datasets['test']

    if args.task_type == 'A_L+A_U->B':
        # Train split (labelled and unlabelled classes) for training
        train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                    unlabelled_dataset=deepcopy(datasets['trainA_unlabelled']))

        unlabelled_train_examples_test = deepcopy(datasets['trainA_unlabelled'])
        unlabelled_train_examples_test.transform = test_transform

        return train_dataset, unlabelled_train_examples_test, datasets
    
    elif args.task_type == 'A_L+A_U+B->A_U+B+C':
        train_dataset = MergedTriDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                        unlabelled_dataset1=deepcopy(datasets['trainA_unlabelled']),
                                        unlabelled_dataset2=deepcopy(datasets['trainB_unlabelled']))

        unlabelled_trainA_test = deepcopy(datasets['trainA_unlabelled'])
        unlabelled_trainA_test.transform = test_transform
        unlabelled_trainB_test = datasets['testB']

        return train_dataset, unlabelled_trainA_test, unlabelled_trainB_test, datasets
    
    else:
        raise NotImplementedError
        

def get_class_splits(args):
    if args.dataset_name in ('cubc', 'scarsc', 'fgvcc'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    if args.dataset_name == 'domainnet':
        args.image_size = 224
        
        if args.pre_splits:
            split_path = os.path.join(osr_split_dir, 'domainnet_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:
            args.train_classes = range(173)
            args.unlabeled_classes = range(173, 345)

    elif args.dataset_name == 'cubc':

        args.image_size = 224

        if use_ssb_splits:
            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif args.dataset_name == 'scarsc':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'fgvcc':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    else:
        raise NotImplementedError

    return args