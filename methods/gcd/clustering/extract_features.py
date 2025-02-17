import torch
from torch.utils.data import DataLoader
import timm
from torchvision import transforms
import torchvision

import argparse
import os
from tqdm import tqdm

from config import domainnet_dataroot
from data.domainnet import DomainNetDataset
from data.augmentations import get_transform
from project_utils.general_utils import strip_state_dict, str2bool
from copy import deepcopy

from config import feature_extract_dir, dino_pretrain_path

parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--root_dir', type=str, default=feature_extract_dir)
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

parser.add_argument('--dataset', type=str, default='domainnet', help='options: cifar10, cifar100, scars')
parser.add_argument('--src_env', type=str)
parser.add_argument('--tgt_env', type=str)
parser.add_argument('--task_type', type=str, default='A_L+A_U->B')

# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda:0')

def extract_features_dino(model, loader, save_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


def extract_features_timm(model, loader, save_dir):

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            images, labels, idxs = batch[:3]
            images = images.to(device)

            features = model.forward_features(images)         # CLS_Token for ViT, Average pooled vector for R50

            # Save features
            for f, t, uq in zip(features, labels, idxs):

                t = t.item()
                uq = uq.item()

                save_path = os.path.join(save_dir, f'{t}', f'{uq}.npy')
                torch.save(f.detach().cpu().numpy(), save_path)


if __name__ == "__main__":
    args.envs = [args.src_env, args.tgt_env]

    args.save_dir = os.path.join('/disk/work/hjwang/gcd/logs/gcd', f'{args.src_env}-{args.tgt_env}')
    args.load_dir = os.path.join('/disk/work/hjwang/gcd/logs/gcd', f'{args.src_env}-{args.tgt_env}', 'dinoB16_best.pt')
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    print('Loading model...')
    # ----------------------
    # MODEL
    # ----------------------
    if args.model_name == 'vit_dino':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path

        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        _, val_transform = get_transform('domainnet', image_size=224, args=args)

    elif args.model_name == 'resnet50_dino':

        extract_features_func = extract_features_dino
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = '/work/whj/pretrained_models/dino/dino_resnet50_pretrain.pth'

        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=False)

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        _, val_transform = get_transform('imagenet', image_size=224, args=args)

    else:

        raise NotImplementedError

    print(f'Using weights from {args.load_dir} ...')
    state_dict = torch.load(args.load_dir)
    model.load_state_dict(state_dict)

    print('Loading data...')
    # ----------------------
    # DATASET
    # ----------------------
    train_dataset = DomainNetDataset(transform=val_transform, root=os.path.join(domainnet_dataroot, args.envs[0]))
    test_dataset  = DomainNetDataset(transform=val_transform, root=os.path.join(domainnet_dataroot, args.envs[1]))
    targets = list(set(train_dataset.targets))

    # ----------------------
    # DATALOADER
    # ----------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Creating base directories...')
    # ----------------------
    # INIT SAVE DIRS
    # Create a directory for each class
    # ----------------------
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for fold in ('train', 'test'):

        fold_dir = os.path.join(args.save_dir, fold)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)

        for t in targets:
            target_dir = os.path.join(fold_dir, f'{t}')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)

    # ----------------------
    # EXTRACT FEATURES
    # ----------------------
    # Extract train features
    train_save_dir = os.path.join(args.save_dir, 'train')
    print('Extracting features from train split...')
    extract_features_func(model=model, loader=train_loader, save_dir=train_save_dir)

    # Extract test features
    test_save_dir = os.path.join(args.save_dir, 'test')
    print('Extracting features from test split...')
    extract_features_func(model=model, loader=test_loader, save_dir=test_save_dir)

    print('Done!')