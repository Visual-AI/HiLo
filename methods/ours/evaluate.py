import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from methods.ours.models.swin_pm import PMTrans, DINOHead
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import str2bool

from config import distortions, severity, ovr_envs, dino_pretrain_path

# Initialize argument parser
parser = argparse.ArgumentParser(description='Evaluation script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset parameters
parser.add_argument('--dataset_name', type=str, default='domainnet')
parser.add_argument('--src_env', type=str)
parser.add_argument('--tgt_env', type=str)
parser.add_argument('--aux_env', type=str, default=None)
parser.add_argument('--task_type', type=str, default='A_L+A_U->B')
parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

# Model parameters
parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
parser.add_argument('--transform', type=str, default='imagenet')

def test(model, test_loader, args):
    """
    Evaluate model on test data
    """
    model.eval()
    
    preds, targets = [], []
    mask = np.array([])
    
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name='Evaluation')
    
    return all_acc, old_acc, new_acc

def main():
    args = parser.parse_args()
    args = get_class_splits(args)
    
    # Set model parameters
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_ctgs = args.num_labeled_classes + args.num_unlabeled_classes
    
    # Initialize model
    model = PMTrans(pretrain_path=dino_pretrain_path, args=args)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")
    
    model.cuda()
    model.eval()
    
    # Get transforms and datasets
    _, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    
    # Initialize test loaders based on task type
    if args.task_type == 'A_L+A_U->B':
        # Get dataset for domain A
        _, unlabeled_dataset_A, _ = get_datasets(args.dataset_name, None, test_transform, args)
        test_loader_A = DataLoader(unlabeled_dataset_A, num_workers=args.num_workers, 
                                 batch_size=args.batch_size, shuffle=False)
        
        print("\nEvaluating on Domain A:")
        all_acc_A, old_acc_A, new_acc_A = test(model, test_loader_A, args)
        print(f'Domain A Accuracies: All {all_acc_A:.4f} | Old {old_acc_A:.4f} | New {new_acc_A:.4f}')
        
    elif args.task_type in ['A_L+A_U+B->A_U+B+C', 'A_L+A_U+B+C->A_U+B+C']:
        # Get datasets for domains A and B
        _, unlabeled_dataset_A, unlabeled_dataset_B, _ = get_datasets(args.dataset_name, None, test_transform, args)
        
        test_loader_A = DataLoader(unlabeled_dataset_A, num_workers=args.num_workers, 
                                 batch_size=args.batch_size, shuffle=False)
        test_loader_B = DataLoader(unlabeled_dataset_B, num_workers=args.num_workers, 
                                 batch_size=args.batch_size, shuffle=False)
        
        print("\nEvaluating on Domain A:")
        all_acc_A, old_acc_A, new_acc_A = test(model, test_loader_A, args)
        print(f'Domain A Accuracies: All {all_acc_A:.4f} | Old {old_acc_A:.4f} | New {new_acc_A:.4f}')
        
        print("\nEvaluating on Domain B:")
        all_acc_B, old_acc_B, new_acc_B = test(model, test_loader_B, args)
        print(f'Domain B Accuracies: All {all_acc_B:.4f} | Old {old_acc_B:.4f} | New {new_acc_B:.4f}')
    
    # Evaluate on domain C (target domains)
    test_loaders = []
    if args.dataset_name == 'domainnet':
        ovr_envs.remove(args.src_env)
        if args.task_type == 'A_L+A_U+B->A_U+B+C':
            ovr_envs.remove(args.aux_env)
            
        for d in ovr_envs:
            args.tgt_env = d
            test_dataset = get_datasets(args.dataset_name, None, test_transform, args)
            test_loader = DataLoader(test_dataset, num_workers=args.num_workers, 
                                   batch_size=args.batch_size, shuffle=False)
            test_loaders.append((d, test_loader))
    else:
        for d in distortions:
            for s in severity:
                args.distortion, args.severity = d, str(s)
                test_dataset = get_datasets(args.dataset_name, None, test_transform, args)
                test_loader = DataLoader(test_dataset, num_workers=args.num_workers, 
                                       batch_size=args.batch_size, shuffle=False)
                test_loaders.append((f"{d}_{s}", test_loader))
    
    print("\nEvaluating on Domain C (target domains):")
    all_accs, old_accs, new_accs = [], [], []
    
    for domain_name, test_loader in test_loaders:
        all_acc, old_acc, new_acc = test(model, test_loader, args)
        print(f'Domain {domain_name} Accuracies: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}')
        all_accs.append(all_acc)
        old_accs.append(old_acc)
        new_accs.append(new_acc)
    
    # Print average results for domain C
    print(f'\nAverage Domain C Accuracies: All {np.mean(all_accs):.4f} | Old {np.mean(old_accs):.4f} | New {np.mean(new_accs):.4f}')

if __name__ == "__main__":
    main() 