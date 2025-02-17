from __future__ import print_function
from collections import OrderedDict
from functools import partial
import numpy as np
import math

from re import A

from timm.models.layers import Mlp, DropPath

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F
from einops import rearrange

from methods.ours.loss import *
from models import vision_transformer as vits

from project_utils.general_utils import finetune_params

from loss import info_nce_logits, SupConLoss


class CrossEntropyMixup(nn.Module):

    def __init__(self, num_classes):
        super(CrossEntropyMixup, self).__init__()
        self.num_classes = num_classes
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, s_lambda=None):

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        targets = targets.to(inputs.device)
        s_lambda = s_lambda.unsqueeze(1)
        targets = s_lambda * targets + (1 - s_lambda) / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        return loss.mean()
    

def cosine_distance(source_hidden_features, target_hidden_features):
    "similarity between different features"
    n_s = source_hidden_features.shape[0]
    n_t = target_hidden_features.shape[0]
    
    temp_matrix = torch.mm(source_hidden_features, target_hidden_features.t())

    for i in range(n_s):
        vec = source_hidden_features[i]
        temp_matrix[i] /= torch.norm(vec, p=2)
    for j in range(n_t):
        vec = target_hidden_features[j]
        temp_matrix[:, j] /= torch.norm(vec, p=2)
    return temp_matrix

def convert_to_onehot(s_label, class_num):
    s_sca_label = s_label.cpu().data.numpy()
    return np.eye(class_num)[s_sca_label]

def mixup_soft_ce(pred, targets, weight, lam):
    """ mixed categorical cross-entropy loss
    """
    loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, targets)
    loss = torch.sum(lam*weight*loss) / (torch.sum(weight*lam).item())
    loss = loss * torch.sum(lam)
    return loss

def mixup_sup_dis(preds, s_label, lam):
    """ mixup_distance_in_feature_space_for_intermediate_source
    """
    label = torch.mm(s_label, s_label.t())
    mixup_loss = -torch.sum(label * F.log_softmax(preds), dim=1)
    mixup_loss = torch.sum (torch.mul(mixup_loss, lam))
    return mixup_loss

def mixup_unsup_dis(preds, lam):
    """ mixup_distance_in_feature_space_for_intermediate_target
    """
    label = torch.eye(preds.shape[0]).cuda()
    mixup_loss = -torch.sum(label* F.log_softmax(preds), dim=1)
    mixup_loss = torch.sum(torch.mul(mixup_loss, lam))
    return mixup_loss

def mix_token(s_token, t_token, s_lambda, t_lambda):
    # print(s_token.shape, s_lambda.shape, t_token.shape)
    s_token = torch.einsum('BNC,BN -> BNC', s_token, s_lambda)
    t_token = torch.einsum('BNC,BN -> BNC', t_token, t_lambda)
    m_tokens = s_token+t_token
    return m_tokens

def mix_lambda_atten(s_scores, t_scores, s_lambda, num_patch):
    t_lambda = 1-s_lambda
    if s_scores is None or t_scores is None:
        s_lambda = torch.sum(s_lambda, dim=1) / num_patch # important for /self.num_patch
        t_lambda = torch.sum(t_lambda, dim=1) / num_patch
        s_lambda = s_lambda / (s_lambda+t_lambda)        
    else:
        s_lambda = torch.sum(torch.mul(s_scores, s_lambda), dim=1) / num_patch # important for /self.num_patch
        t_lambda = torch.sum(torch.mul(t_scores, t_lambda), dim=1) / num_patch
        s_lambda = s_lambda / (s_lambda+t_lambda)
    return s_lambda


def mix_lambda (s_lambda,t_lambda):
    return torch.sum(s_lambda,dim=1) / (torch.sum(s_lambda,dim=1) + torch.sum(t_lambda,dim=1))


def softplus(x):
    return  torch.log(1+torch.exp(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        save = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, save 


class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        t, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(t)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn     


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits
    

class PMTrans(nn.Module):
    '''
    Modified from the original PMTrans (https://arxiv.org/abs/2303.13434) 
    '''
    def __init__(self, pretrain_path, args):
        super(PMTrans, self).__init__()

        self.backbone = vits.__dict__['vit_base']()
        state_dict = torch.load(pretrain_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict)
        finetune_params(self.backbone, args) # HOW MUCH OF BASE MODEL TO FINETUNE

        self.num_patch = 196

        self.sem_head = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs)
        self.s_dist_alpha = nn.Parameter(torch.Tensor([1]))
        self.s_dist_beta = nn.Parameter(torch.Tensor([1]))
        self.super_ratio = nn.Parameter(torch.Tensor([-2]))
        self.unsuper_ratio = nn.Parameter(torch.Tensor([-2]))

        self.sup_ce_loss = CrossEntropyMixup(num_classes=args.num_ctgs)
 
    def attn_map(self, attn=None):
        scores = attn
            
        n_p_e = int(np.sqrt(self.num_patch))
        n_p_f = int(np.sqrt(scores.size(1)))

        scores = F.interpolate(rearrange(scores, 'B (H W) -> B 1 H W', H = n_p_f), size=(n_p_e, n_p_e)).squeeze(1)
        scores = rearrange(scores, 'B H W -> B (H W)')
        return scores.softmax(dim=-1)

    def mix_source_target(self, s_token, t_token, min_length, s_lambda, t_lambda, pred, infer_label, s_feats, t_feats, s_scores, t_scores, weight_tgt, weight_src, cluster_criterion=None, epoch=None, args=None):
        tok_num, tok_dims = s_token.shape[1], s_token.shape[2]
        feat_dims = s_feats.shape[-1]
        
        s_token = s_token.view(s_token.shape[0]//2, 2, tok_num, tok_dims)[:min_length//2].view(min_length, tok_num, tok_dims)
        t_token = t_token.view(t_token.shape[0]//2, 2, tok_num, tok_dims)[:min_length//2].view(min_length, tok_num, tok_dims)
        pred = pred.view(pred.shape[0]//2, 2)[:min_length//2].view(min_length)
        infer_label = infer_label.view(infer_label.shape[0]//2, 2)[:min_length//2].view(min_length)
        s_feats = s_feats.view(s_feats.shape[0]//2, 2, feat_dims)[:min_length//2].view(min_length, feat_dims)
        t_feats = t_feats.view(t_feats.shape[0]//2, 2, feat_dims)[:min_length//2].view(min_length, feat_dims)
        s_scores = s_scores.view(s_scores.shape[0]//2, 2, tok_num)[:min_length//2].view(min_length, tok_num)
        weight_tgt = weight_tgt.view(weight_tgt.shape[0]//2, 2)[:min_length//2].view(min_length)
        weight_src = weight_src.reshape(weight_src.shape[0], 2, weight_src.shape[-1]//2)[:, :, :min_length//2].reshape(min_length)
        s_lambda = s_lambda.unsqueeze(0).expand(2, -1, -1).reshape(min_length, self.num_patch)
        t_lambda = t_lambda.unsqueeze(0).expand(2, -1, -1).reshape(min_length, self.num_patch)

        m_s_t_tokens = mix_token(s_token, t_token, s_lambda, t_lambda)        
        # m_s_t_tokens = mix_token(s_token[:min_length], t_token[:min_length], s_lambda, t_lambda)        
        m_s_t_feats, _ = self.backbone.forward_features(m_s_t_tokens, patch=True)
        m_s_t_feats, m_s_t_logits = self.sem_head(m_s_t_feats)
        t_scores = (torch.ones(m_s_t_feats.size(0), tok_num) / tok_num).cuda()
        s_lambda = mix_lambda_atten(s_scores, t_scores, s_lambda, tok_num) # with attention map
        t_lambda = 1 - s_lambda

        # represent learning for mixup, sup
        m_s_t_feats1 = torch.cat([f.unsqueeze(1) for f in m_s_t_feats.chunk(2)], dim=1)
        m_s_t_feats1 = torch.nn.functional.normalize(m_s_t_feats1, dim=-1)
        feature_space_loss = SupConLoss()(m_s_t_feats1, labels=infer_label[:min_length//2], s_lambda=s_lambda)

        # represent learning for mixup, unsup
        contrastive_logits, contrastive_labels = info_nce_logits(features=m_s_t_feats)
        unsup_con_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
        feature_space_loss += unsup_con_loss

        # clustering for mixup, sup
        label_space_loss = self.sup_ce_loss(m_s_t_logits, infer_label, s_lambda)

        # clustering for mixup, unsup
        teacher_out = m_s_t_logits.detach()
        cluster_loss = cluster_criterion(m_s_t_logits, teacher_out, epoch)
        avg_probs = m_s_t_logits.softmax(dim=1).mean(dim=0)
        me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
        cluster_loss += args.memax_weight * me_max_loss
        label_space_loss += cluster_loss

        return feature_space_loss, label_space_loss


    def forward(self, mixture, mask_lab=None, infer_label=None, mem_fea=None, mem_cls=None, img_idx=None, class_weight_src=None):
        device = mixture.device

        mixed_tokens = self.backbone.patch_embed(mixture)
        student_proj, mixed_attn = self.backbone.forward_features(mixed_tokens, patch=True)
        student_proj, student_out = self.sem_head(student_proj)

        if self.training:
            s_token = torch.cat([f[mask_lab] for f in mixed_tokens.chunk(2)], dim=0)
            s_feats = torch.cat([f[mask_lab] for f in student_proj.chunk(2)], dim=0)
            s_attn = torch.cat([f[mask_lab] for f in mixed_attn.chunk(2)], dim=0)
            s_scores = self.attn_map(attn=s_attn)

            t_token = torch.cat([f[~mask_lab] for f in mixed_tokens.chunk(2)], dim=0)
            t_feats = torch.cat([f[~mask_lab] for f in student_proj.chunk(2)], dim=0)
            t_attn = torch.cat([f[~mask_lab] for f in mixed_attn.chunk(2)], dim=0)
            t_pred = torch.cat([f[~mask_lab] for f in student_out.chunk(2)], dim=0)
            t_cls = t_pred.softmax(dim=-1)
            
            dis = -torch.mm(t_feats.detach(), mem_fea.t())
            for di in range(dis.size(0)):
                dis[di, img_idx[di]] = torch.max(dis)
            _, p1 = torch.sort(dis, dim=1)
            w = torch.zeros(t_feats.size(0), mem_fea.size(0)).cuda()
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p1[wi, wj]] = 1 / 5
            weight_tgt, pred = torch.max(w.mm(mem_cls), 1)

            weight_src = class_weight_src[infer_label].unsqueeze(0)

            t_scores = self.attn_map(attn=t_attn)

            min_length = min(s_token.shape[0], t_token.shape[0])
            t_lambda = dists.Beta(softplus(self.s_dist_alpha), softplus(self.s_dist_beta)).rsample((min_length//2, self.num_patch,)).to(device).squeeze(-1)
            s_lambda = 1 - t_lambda

            super_m_s_t_loss, unsuper_m_s_t_loss = self.mix_source_target(s_token, t_token, min_length, s_lambda, t_lambda, pred, infer_label, s_feats, t_feats, s_scores, t_scores, weight_tgt, weight_src)
            total_loss = softplus(self.super_ratio) * super_m_s_t_loss + softplus(self.unsuper_ratio) * unsuper_m_s_t_loss
            
            return total_loss, student_proj, student_out, t_cls, t_feats, weight_src
        else:
            return student_proj, student_out
        

    def forward_features(self, mixture, mask_lab=None, infer_label=None, mem_fea=None, mem_cls=None, img_idx=None, class_weight_src=None, cluster_criterion=None, epoch=None, args=None):
        device = mixture.device

        mixed_tokens = self.backbone.patch_embed(mixture)
        feats_list, mixed_attn = self.backbone.forward_features(mixed_tokens, patch=True, nth_layers=[1, 12])
        domain_feats = feats_list[0][:, 0]
        semantic_feats = feats_list[1][:, 0]
        del feats_list
        student_proj, student_out = self.sem_head(semantic_feats)

        if self.training:
            s_token = torch.cat([f[mask_lab] for f in mixed_tokens.chunk(2)], dim=0)
            s_feats = torch.cat([f[mask_lab] for f in student_proj.chunk(2)], dim=0)
            s_attn = torch.cat([f[mask_lab] for f in mixed_attn.chunk(2)], dim=0)
            s_scores = self.attn_map(attn=s_attn)

            t_token = torch.cat([f[~mask_lab] for f in mixed_tokens.chunk(2)], dim=0)
            t_feats = torch.cat([f[~mask_lab] for f in student_proj.chunk(2)], dim=0)
            t_attn = torch.cat([f[~mask_lab] for f in mixed_attn.chunk(2)], dim=0)
            t_pred = torch.cat([f[~mask_lab] for f in student_out.chunk(2)], dim=0)
            t_cls = t_pred.softmax(dim=-1)
            
            dis = -torch.mm(t_feats.detach(), mem_fea.t())
            for di in range(dis.size(0)):
                dis[di, img_idx[di]] = torch.max(dis)
            _, p1 = torch.sort(dis, dim=1)
            w = torch.zeros(t_feats.size(0), mem_fea.size(0)).cuda()
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p1[wi, wj]] = 1 / 5
            weight_tgt, pred = torch.max(w.mm(mem_cls), 1)

            weight_src = class_weight_src[infer_label].unsqueeze(0)

            t_scores = self.attn_map(attn=t_attn)

            min_length = min(s_token.shape[0], t_token.shape[0])
            t_lambda = dists.Beta(softplus(self.s_dist_alpha), softplus(self.s_dist_beta)).rsample((min_length//2, self.num_patch,)).to(device).squeeze(-1)
            s_lambda = 1 - t_lambda

            super_m_s_t_loss, unsuper_m_s_t_loss = self.mix_source_target(s_token, t_token, min_length, s_lambda, t_lambda, pred, infer_label, s_feats, t_feats, s_scores, t_scores, weight_tgt, weight_src, cluster_criterion, epoch, args)
            total_loss = softplus(self.super_ratio) * super_m_s_t_loss + softplus(self.unsuper_ratio) * unsuper_m_s_t_loss
            
            return total_loss, semantic_feats, domain_feats, t_cls, t_feats
        else:
            return student_proj, student_out