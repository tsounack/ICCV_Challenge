import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import lambertw


class SuperLoss(nn.Module):
    def __init__(self, C, lam=0.25):
        super(SuperLoss, self).__init__()
        self.tau = torch.log(torch.FloatTensor([C])).cuda()
        self.lam = torch.FloatTensor([lam]).cuda()

    def forward(self, l_i):
        l_i_detach = l_i.detach()
        # self.tau = 0.9 * self.tau + 0.1 * l_i_detach
        sigma = self.sigma(l_i_detach)
        loss = (l_i - self.tau) * sigma + self.lam * torch.log(sigma) ** 2
        loss = loss.mean()
        return loss

    def sigma(self, l_i):
        x = -2 / torch.exp(torch.ones_like(l_i))
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma
        

class WeightedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(WeightedBCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)
    def forward(self, output, target):
        # modify weights based on training data occurences
        # only works for mimic dataset!
        label_freqs = np.array((1980.0, 54954.0, 1004.0, 4366.0, 12905.0, 33507.0, 635.0, \
                                38579.0, 3465.0, 61515.0, 54002.0, 6161.0, 24074.0, 468.0, \
                                30799.0, 9488.0, 8276.0, 63845.0, 2756.0, 12039.0, 71300.0, \
                                7591.0, 3387.0, 534.0, 3258.0, 2806.0, 1972.0))
        dataset_size = 211879
        negative_freqs = dataset_size - label_freqs
        weights = negative_freqs / label_freqs
        self.pos_weight = torch.tensor(weights).cuda()
        return super(WeightedBCEWithLogitsLoss, self).forward(output, target)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean', **kwargs):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, output, target):
        B, c = output.size()
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.smoothing / c + (1 - self.smoothing) * F.nll_loss(log_preds, target,
                                                                             reduction=self.reduction)


class LabelSmoothingCrossEntropyWithSuperLoss(nn.Module):
    def __init__(self, classes, eps=0.1, reduction='mean', **kwargs):
        super(LabelSmoothingCrossEntropyWithSuperLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.classes = classes
        self.super_loss = SuperLoss(C=classes)

    def forward(self, output, target):
        B, c = output.size()
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        loss_cls = loss * self.eps / c + (1 - self.eps) * self.super_loss(
            F.nll_loss(log_preds, target, reduction='none'))
        return loss_cls

    def __repr__(self):
        s = 'LabelSmoothingCrossEntropyWithSuperLoss (eps=' + str(
            self.eps) + ', reduction=' + self.reduction + ', classes=' + str(self.classes) + ')'
        return s


class MixUpLoss(nn.Module):
    def __init__(self, criterion, **kwargs):
        super().__init__()
        self.criterion = MixUpLoss.get_loss(criterion, **kwargs)

    def forward(self, pred, label, label_mixed, lam):
        label_mixed = label_mixed.cuda()
        lam = lam.cuda()
        return lam * self.criterion(pred, label) + (1 - lam) * self.criterion(pred, label_mixed)

    def __repr__(self):
        s = 'MixUpLoss (criterion=' + str(self.criterion) + ')'
        return s

    @staticmethod
    def get_loss(loss_func, **loss_args):
        if hasattr(nn, loss_func):
            loss_func = getattr(nn, loss_func)(**loss_args)
        else:
            try:
                loss_func = eval(loss_func)(**loss_args)
            except NameError:
                raise NotImplementedError("Loss {} not implemented".format(loss_args))
        return loss_func
