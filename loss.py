import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from project_utils.loss_utils import WarmStartGradientReverseLayer

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, s_lambda=None, weights=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        if s_lambda is not None:
            mask = mask.repeat(anchor_count, 1)
            mask = mask * s_lambda[:, None]
            mask = mask.repeat(1, contrast_count)
            dig = torch.eye(batch_size*2, dtype=torch.float32).to(device)
            logits_mask = 1 - dig

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (logits_mask * log_prob).sum(1) / logits_mask.sum(1)

        else:
            mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )
            
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if weights is not None:
            loss = torch.sum(weights.view(anchor_count, batch_size) * loss.view(anchor_count, batch_size)) / (2*torch.sum(weights).item())
        else:
            loss = loss.view(anchor_count, batch_size).mean()

        return loss


def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class MarginLoss(nn.Module):
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()

        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, feat_src, feat_tgt):
        """
        Compute Gram matrix
        Args:
            feat_src: batch_size_src * feature_dim
            feat_tgt: batch_size_tat * feature_dim
            
        Returns: 
            Matrix form of (batch_size_src + batch_size_tat) * (batch_size_src + batch_size_tat):
            [   K_ss K_st
                K_ts K_tt ]
        """
        n, m = feat_src.size(0), feat_tgt.size(0)
        total = torch.cat([feat_src, feat_tgt], dim=0)
        feat_dim = total.size(1)

        total0 = total.unsqueeze(0).expand(n+m, n+m, feat_dim) # each data is expanded into (n+m) copies.
        total1 = total.unsqueeze(1).expand(n+m, n+m, feat_dim) # each row of data is expanded into (n+m) copies.
        L2_distance = ((total0 - total1)**2).sum(2) # compute |x-y| in gaussian kernel

        # bandwidth for each kernel
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / ((n + m)**2 - (n + m))

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]

        # exp(-|x-y|/bandwith) in gaussian kernel
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val) # combine multiple kernels

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        """        
            Args:
                source: batch_size_src * feature_dim
                target: batch_size_tat * feature_dim
        """
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(source, target)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class NCMMDLoss(nn.Module):
    def __init__(self, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, gamma=1.0, max_iter=1000, **kwargs):
        '''
        Args:
            kernel_mul: with bandwidth as the center, and the base on both sides expanding.
            kernel_num: num of kernels
            fix_sigma: whether use a fixed sigma, using single kernel if fixed.
        '''
        super(NCMMDLoss, self).__init__()
        self.num_class = num_class
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def guassian_kernel(self, feat_src, feat_tgt):
        """
        Compute Gram matrix
        Args:
            feat_src: batch_size_src * feature_dim
            feat_tgt: batch_size_tat * feature_dim
            
        Returns: 
            Matrix form of (batch_size_src + batch_size_tat) * (batch_size_src + batch_size_tat):
            [   K_ss K_st
                K_ts K_tt ]
        """
        n, m = feat_src.size(0), feat_tgt.size(0)
        total = torch.cat([feat_src, feat_tgt], dim=0)
        feat_dim = total.size(1)

        total0 = total.unsqueeze(0).expand(n+m, n+m, feat_dim) # each data is expanded into (n+m) copies.
        total1 = total.unsqueeze(1).expand(n+m, n+m, feat_dim) # each row of data is expanded into (n+m) copies.
        L2_distance = ((total0 - total1)**2).sum(2) # compute |x-y| in gaussian kernel

        # bandwidth for each kernel
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / ((n + m)**2 - (n + m))

        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]

        # exp(-|x-y|/bandwith) in gaussian kernel
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val) # combine multiple kernels
    
    def cal_weight(self, label_src, logits_tgt):
        batch_size = label_src.size()[0]
        label_src = label_src.cpu().data.numpy()
        label_src_onehot = np.eye(self.num_class)[label_src] # one hot

        label_src_sum = np.sum(label_src_onehot, axis=0, keepdims=True)
        label_src_sum[label_src_sum == 0] = 1.0
        label_src_onehot = label_src_onehot / label_src_sum # label ratio

        # Pseudo label
        target_label = logits_tgt.cpu().data.max(1)[1].numpy()

        logits_tgt = logits_tgt.cpu().data.numpy()
        target_logits_sum = np.sum(logits_tgt, axis=0, keepdims=True)
        target_logits_sum[target_logits_sum == 0] = 1.0
        logits_tgt = logits_tgt / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(label_src)
        set_t = set(target_label)
        count = 0

        for i in range(self.num_class): # (B, C)
            if i in set_s and i in set_t:
                s_tvec = label_src_onehot[:, i].reshape(batch_size, -1) # (B, 1)
                t_tvec = logits_tgt[:, i].reshape(batch_size, -1) # (B, 1)
                
                ss = np.dot(s_tvec, s_tvec.T) # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st     
                count += 1

        weight_ss = weight_ss / count if count != 0 else np.array([0])
        weight_tt = weight_tt / count if count != 0 else np.array([0])
        weight_st = weight_st / count if count != 0 else np.array([0])

        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

    def forward(self, feat_src, feat_tgt, label_src, logits_tgt):
        batch_size = feat_src.size(0)

        weight_ss, weight_tt, weight_st = self.cal_weight(label_src, logits_tgt)
        weight_ss = torch.from_numpy(weight_ss).to(feat_src.device) # B, B
        weight_tt = torch.from_numpy(weight_tt).to(feat_src.device)
        weight_st = torch.from_numpy(weight_st).to(feat_src.device)

        kernels = self.guassian_kernel(feat_src, feat_tgt)
        
        if torch.sum(torch.isnan(sum(kernels))):
            return torch.Tensor([0]).to(feat_src.device)

        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)

        print(loss.shape)
        # Dynamic weighting
        lamb = 2. / (1. + np.exp(-self.gamma * self.curr_iter / self.max_iter)) - 1
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)
        loss = loss * lamb

        return loss


def DJSLoss(scores, cls_mask, device=torch.device('cuda')):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    
    scores = scores.to(device)
    cls_mask = cls_mask.to(device)

    n, m = scores.shape

    scores_pos_pair = scores[cls_mask]
    nn = scores_pos_pair.size(0)

    first_term = -F.softplus(-scores_pos_pair).mean()
    second_term = (torch.sum(F.softplus(scores)) - torch.sum(F.softplus(scores_pos_pair))) / (n * m - nn)
    return first_term - second_term


class MIloss(nn.Module):
    """Loss function to extract semantic information and substract exclusive information from images, using mutual information"""

    def __init__(self, device):
        super(MIloss, self).__init__()
        self.device = device

    def _get_class_mask(self, labels_first, labels_second):
        labels = torch.cat([labels_first, labels_second], dim=0)
        labels_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        # discard the main diagonal from both: labels and similarities matrix
        diag_mask = torch.eye(labels_mask.shape[0], dtype=torch.bool)

        return labels_mask.detach(), diag_mask.detach()

    def _partition_matrix(self, matrix, batch_size_first):
        ul = matrix[:batch_size_first, :batch_size_first]
        ur = matrix[:batch_size_first, batch_size_first:]
        ll = matrix[batch_size_first:, :batch_size_first]
        lr = matrix[batch_size_first:, batch_size_first:]
        return ul, ur, ll, lr

    def _mask_matrix(self, partitioned_matrix_ls, diag_ul, diag_lr, batch_size_first, batch_size_second):
        partitioned_matrix_ls[0] = partitioned_matrix_ls[0][~diag_ul].view(batch_size_first, -1)
        partitioned_matrix_ls[1] = partitioned_matrix_ls[1].view(batch_size_first, -1)
        partitioned_matrix_ls[2] = partitioned_matrix_ls[2].view(batch_size_second, -1)
        partitioned_matrix_ls[3] = partitioned_matrix_ls[3][~diag_lr].view(batch_size_second, -1)
        return partitioned_matrix_ls

    def partition_and_mask_matrix(self, similarity_matrix, labels_first, labels_second):
        batch_size_first = labels_first.size(0)
        batch_size_second = labels_second.size(0)

        labels_mask, diag_mask = self._get_class_mask(labels_first, labels_second)

        sm_ul, sm_ur, sm_ll, sm_lr = self._partition_matrix(similarity_matrix, batch_size_first)
        diag_ul, _, _, diag_lr = self._partition_matrix(diag_mask, batch_size_first)
        lb_ul, lb_ur, lb_ll, lb_lr = self._partition_matrix(labels_mask, batch_size_first)

        partitioned_sm_ls = self._mask_matrix([sm_ul, sm_ur, sm_ll, sm_lr], diag_ul, diag_lr, batch_size_first, batch_size_second)
        partitioned_lb_ls = self._mask_matrix([lb_ul, lb_ur, lb_ll, lb_lr], diag_ul, diag_lr, batch_size_first, batch_size_second)

        return partitioned_sm_ls, partitioned_lb_ls


    def forward(self, shallow_feat, deep_feat, labels_A, labels_B, mask_lab, ws_list):
        deep_feat_A = torch.cat([f[mask_lab] for f in deep_feat.chunk(2)], dim=0)
        deep_feat_B = torch.cat([f[~mask_lab] for f in deep_feat.chunk(2)], dim=0)
        shallow_feat_A = torch.cat([f[mask_lab] for f in shallow_feat.chunk(2)], dim=0)
        shallow_feat_B = torch.cat([f[~mask_lab] for f in shallow_feat.chunk(2)], dim=0)

        dAsB = torch.cat([deep_feat_A, shallow_feat_B], dim=0)
        dBsA = torch.cat([deep_feat_B, shallow_feat_A], dim=0)
        dAsA = torch.cat([deep_feat_A, shallow_feat_A], dim=0)
        dBsB = torch.cat([deep_feat_B, shallow_feat_B], dim=0)

        dAsB = F.normalize(dAsB, dim=1)
        dBsA = F.normalize(dBsA, dim=1)
        dAsA = F.normalize(dAsA, dim=1)
        dBsB = F.normalize(dBsB, dim=1)

        similarity_matrix1 = torch.matmul(dAsB, dAsB.T)
        similarity_matrix2 = torch.matmul(dBsA, dBsA.T)
        similarity_matrix3 = torch.matmul(dAsA, dAsA.T)
        similarity_matrix4 = torch.matmul(dBsB, dBsB.T)

        partitioned_sm_ls1, partitioned_lb_ls1 = self.partition_and_mask_matrix(similarity_matrix1, labels_A, labels_B)
        partitioned_sm_ls2, partitioned_lb_ls2 = self.partition_and_mask_matrix(similarity_matrix2, labels_B, labels_A)
        partitioned_sm_ls3, partitioned_lb_ls3 = self.partition_and_mask_matrix(similarity_matrix3, labels_A, labels_A)
        partitioned_sm_ls4, partitioned_lb_ls4 = self.partition_and_mask_matrix(similarity_matrix4, labels_B, labels_B)
        
        del similarity_matrix1, similarity_matrix2, similarity_matrix3, similarity_matrix4

        loss1 = ws_list[0] * DJSLoss(partitioned_sm_ls1[0], partitioned_lb_ls1[0]) + ws_list[1] * DJSLoss(partitioned_sm_ls1[-1], partitioned_lb_ls1[-1])
        loss1 += ws_list[2] * DJSLoss(partitioned_sm_ls2[0], partitioned_lb_ls2[0]) + ws_list[3] * DJSLoss(partitioned_sm_ls2[-1], partitioned_lb_ls2[-1])
        loss2 = ws_list[4] * DJSLoss(partitioned_sm_ls1[1], partitioned_lb_ls1[1]) + ws_list[5] * DJSLoss(partitioned_sm_ls1[2], partitioned_lb_ls1[2])
        loss2 += ws_list[6] * DJSLoss(partitioned_sm_ls2[1], partitioned_lb_ls2[1]) + ws_list[7] * DJSLoss(partitioned_sm_ls2[2], partitioned_lb_ls2[2])
        loss3 = ws_list[8] * DJSLoss(partitioned_sm_ls3[1], partitioned_lb_ls3[1]) + ws_list[9] * DJSLoss(partitioned_sm_ls3[2], partitioned_lb_ls3[2])
        loss4 = ws_list[10] * DJSLoss(partitioned_sm_ls4[1], partitioned_lb_ls4[1]) + ws_list[11] * DJSLoss(partitioned_sm_ls4[2], partitioned_lb_ls4[2])

        loss = loss1 + loss2 + loss3 + loss4
        
        del partitioned_sm_ls1, partitioned_lb_ls1, partitioned_sm_ls2, partitioned_lb_ls2
        
        return loss


class Distangleloss(nn.Module):
    """Loss function to extract semantic information and substract exclusive information from images, using mutual information"""

    def __init__(self, device):
        super(Distangleloss, self).__init__()
        self.device = device

    def js_fgan_lower_bound(self, score_matrix):
        """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
        f_diag = score_matrix.diag()
        first_term = -F.softplus(-f_diag).mean()
        n = score_matrix.size(0)
        second_term = (torch.sum(F.softplus(score_matrix)) -
                    torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
        return first_term - second_term

    def forward(self, mlp_head, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        _, scores = mlp_head(xy_pairs)
        similarity_matrix = torch.reshape(scores, [batch_size, batch_size]).t()
        loss = self.js_fgan_lower_bound(similarity_matrix)
        
        return loss


class MCC_DALN(nn.Module):
    def __init__(self, mlp_head, device, args):
        super(MCC_DALN, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.mlp_head = mlp_head
        self.device = device
        self.args = args

    def _entropy(self, x):
        bs = x.size(0)
        epsilon = 1e-5
        entropy = -x * torch.log(x + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def compute_mcc_loss(self, logits, mask_lab, temperature=2.5):
        outputs_target = torch.cat([f[~mask_lab] for f in logits.chunk(2)], dim=0)

        outputs_target_temp = outputs_target / temperature
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)

        target_entropy_weight = self._entropy(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = self.args.batch_size * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1,1)).transpose(1,0).mm(target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / self.args.num_ctgs

        return mcc_loss

    @staticmethod
    def n_discrepancy(y_s, y_t):
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = (-torch.norm(pre_t, 'nuc') + torch.norm(pre_s, 'nuc')) / y_t.shape[0]
        return loss

    def compute_daln_loss(self, features, mask_lab):
        f_grl = self.grl(features)
        _, logits = self.mlp_head(f_grl)
        y_s = torch.cat([f[mask_lab] for f in logits.chunk(2)], dim=0)
        y_t = torch.cat([f[~mask_lab] for f in logits.chunk(2)], dim=0)

        daln_loss = self.n_discrepancy(y_s, y_t)
        
        return daln_loss

    def forward(self, features, logits, mask_lab):
        mcc_loss = self.compute_mcc_loss(logits, mask_lab)
        daln_loss = self.compute_daln_loss(features, mask_lab)

        loss = mcc_loss - daln_loss

        return loss