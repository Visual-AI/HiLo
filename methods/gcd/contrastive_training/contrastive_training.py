import argparse
import os
import numpy as np
from tqdm import tqdm

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader

from models import vision_transformer as vits
from models import clip_vit as clip_vit

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from project_utils.cluster_and_log_utils import log_accs_from_preds
from project_utils.general_utils import str2bool, AverageMeter, finetune_params

from loss import info_nce_logits, SupConLoss, ContrastiveLearningViewGenerator

from config import clip_pretrain_path, distortions, severity

parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=16, type=int)

parser.add_argument('--warmup_model_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')

parser.add_argument('--dataset_name', type=str, default='domainnet', help='options: [domainnet, cubc]')
parser.add_argument('--model_path', type=str)
parser.add_argument('--src_env', type=str)
parser.add_argument('--tgt_env', type=str)
parser.add_argument('--aux_env', type=str, default=None)
parser.add_argument('--task_type', type=str, default='A_L+A_U->B')
parser.add_argument('--loss', type=str, default='baseline')
parser.add_argument('--prop_train_labels', type=float, default=0.5)
parser.add_argument('--pre_splits', type=str2bool, default=False)
parser.add_argument('--use_ssb_splits', type=str2bool, default=True)
parser.add_argument('--only_test', type=str2bool, default=False)
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

parser.add_argument('--grad_from_block', type=int, default=11)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--save_best_thresh', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--transform', type=str, default='domainnet')
parser.add_argument('--seed', default=1, type=int)

parser.add_argument('--model', type=str, default='dino')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--sup_con_weight', type=float, default=0.5)
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda:0')
args = get_class_splits(args)


def train(model, train_loader, optimizer, exp_lr_scheduler, sup_con_crit, args):

    loss_record = AverageMeter()
    train_acc_record = AverageMeter()

    model.train()

    for batch_idx, batch in enumerate(tqdm(train_loader)):

        images, class_labels, uq_idxs, mask_lab = batch
        mask_lab = mask_lab[:, 0]

        class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
        images = torch.cat(images, dim=0).cuda(non_blocking=True)

        # Extract features with base model
        features = model(images)

        # L2-normalize features
        features = torch.nn.functional.normalize(features, dim=-1)

        # Choose which instances to run the contrastive loss on
        if args.contrast_unlabel_only:
            # Contrastive loss only on unlabelled instances
            f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
            con_feats = torch.cat([f1, f2], dim=0)
        else:
            # Contrastive loss for all examples
            con_feats = features

        contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats)
        contrastive_loss = nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

        # Supervised contrastive loss
        f1, f2 = [f[mask_lab] for f in features.chunk(2)]
        sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        sup_con_labels = class_labels[mask_lab]

        sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

        # Total loss
        loss = (1 - args.sup_con_weight) * contrastive_loss + args.sup_con_weight * sup_con_loss

        # Train acc
        _, pred = contrastive_logits.max(1)
        acc = (pred == contrastive_labels).float().mean().item()
        train_acc_record.update(acc, pred.size(0))

        loss_record.update(loss.item(), class_labels.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step schedule
    exp_lr_scheduler.step()


def test_kmeans(model, test_loader, epoch, save_name, args):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    # print('Done!')
    del kmeans
    
    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name)

    del all_feats, targets, mask

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # Hyper-paramters
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.mlp_out_dim = 65536

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.model == 'dino':
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    elif args.model == 'clip':
        model_clip = clip_vit.load_clip(clip_pretrain_path)
        backbone = model_clip.visual.float().to(device)

    finetune_params(backbone, args) # HOW MUCH OF BASE MODEL TO FINETUNE

    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    projection_head.to(device)

    model = nn.Sequential(backbone, projection_head).to(device)

    # ----------------------
    # OPTIMIZATION
    # ----------------------
    optimizer = SGD(list(projection_head.parameters()) + list(backbone.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss()

    # --------------------
    # TRANSFORMS
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    if args.aux_env is None:
        args.envs = [args.src_env, args.tgt_env]
    else:
        args.envs = [args.src_env, args.aux_env, args.tgt_env]

    if args.task_type == 'A_L+A_U->B':
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    elif args.task_type == 'A_L+A_U+B->A_U+B+C':
        train_dataset, test_dataset, unlabelled_train_examples_test, unlabelled_train_examples_test2, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=False)    
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=False)
    if args.task_type == 'A_L+A_U+B->A_U+B+C':                                  
        test_loader_unlabelled2 = DataLoader(unlabelled_train_examples_test2, num_workers=args.num_workers,
                                            batch_size=256, shuffle=False, pin_memory=False)
    # ----------------------
    # TRAIN
    # ----------------------
    best_test_acc, best_train_ul_acc, best_train2_ul_acc = 0, 0, 0

    for epoch in range(args.epochs):
        print("Epoch: " + str(epoch))

        train(model, train_loader, optimizer, exp_lr_scheduler, sup_con_crit, args)

        with torch.no_grad():
            # print('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test_kmeans(backbone, test_loader_unlabelled, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
            
            if all_acc > best_train_ul_acc:
                print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
                torch.save(backbone.state_dict(), os.path.join(args.model_path, 'dinoB16_best_trainul.pt'))
                torch.save(projection_head.state_dict(), os.path.join(args.model_path, 'dinoB16_proj_head_best_trainul.pt'))
                best_train_ul_acc = all_acc

                if args.task_type == 'A_L+A_U+B->A_U+B+C':                                  
                    all_acc, old_acc, new_acc = test_kmeans(backbone, test_loader_unlabelled2, epoch=epoch, save_name='Train ACC Unlabelled2', args=args)
                    if all_acc > best_train2_ul_acc:
                        print('Best Train-2 Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
                        torch.save(backbone.state_dict(), os.path.join(args.model_path, 'dinoB16_best_trainul2.pt'))
                        torch.save(projection_head.state_dict(), os.path.join(args.model_path, 'dinoB16_proj_head_best_trainul2.pt'))
                        best_train2_ul_acc = all_acc

            # state_dict = torch.load(os.path.join(args.model_path, 'dinoB16_best_trainul.pt'), map_location='cpu')
            # head_state_dict = torch.load(os.path.join(args.model_path, 'dinoB16_proj_head_best_trainul.pt'), map_location='cpu')
            # backbone.load_state_dict(state_dict)
            # projection_head.load_state_dict(head_state_dict)
            
            if args.dataset_name == 'domainnet':
                # print('Testing on disjoint test set...')
                all_acc_test, old_acc_test, new_acc_test = test_kmeans(backbone, test_loader_labelled, epoch=epoch, save_name='Test ACC', args=args)

            else:
                args.only_test = True
                all_acc_test_arr, old_acc_test_arr, new_acc_test_arr = [], [], []

                for d in distortions:
                    for s in severity:
                        args.distortion, args.severity = d, str(s)
                        test_dataset = get_datasets(args.dataset_name, train_transform, test_transform, args)
                        test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=False)
                        # print('Testing on disjoint test set...')
                        tmp_all_acc_test, tmp_old_acc_test, tmp_new_acc_test = test_kmeans(backbone, test_loader_labelled, epoch=epoch, save_name='Test ACC', args=args)
                        all_acc_test_arr.append(tmp_all_acc_test)
                        old_acc_test_arr.append(tmp_old_acc_test)
                        new_acc_test_arr.append(tmp_new_acc_test)

                all_acc_test, old_acc_test, new_acc_test = sum(all_acc_test_arr)/(len(distortions)*len(severity)), sum(old_acc_test_arr)/(len(distortions)*len(severity)), sum(new_acc_test_arr)/(len(distortions)*len(severity))

            if all_acc_test > best_test_acc:
                print('Best Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

                if not args.dataset_name == 'domainnet':
                    col = len(severity)
                    for i, d in enumerate(distortions):
                        print('For '+d+', All {:.4f} | Old {:.4f} | New {:.4f}'.format(sum(all_acc_test_arr[i*col:(i+1)*col])/col, sum(old_acc_test_arr[i*col:(i+1)*col])/col, sum(new_acc_test_arr[i*col:(i+1)*col])/col))
                    print('################################')
                            
                torch.save(backbone.state_dict(), os.path.join(args.model_path, 'dinoB16_best.pt'))
                # print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

                torch.save(projection_head.state_dict(), os.path.join(args.model_path, 'dinoB16_proj_head_best.pt'))
                # print("projection head saved to {}.".format(args.model_path[:-3] + f'_proj_head_best.pt'))

                best_test_acc = all_acc_test
