import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        # print("sim",sim)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))#256->128
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel# seemed useless

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)


class DiceLoss(nn.Module):
    """
    Implementation of mean soft-dice loss for semantic segmentation
    """
    __EPSILON = 1e-6

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
        y_pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        y_true: a tensor of shape [B, H, W].
        Returns:
        float: soft-iou loss.
        """
        # Convert 3d masks to 2d
        if len(y_pred.shape) == 5:
            y_pred = y_pred.view(*y_pred.shape[:-2], -1)
            y_true = y_true.view(*y_true.shape[:-2], -1)

        num_classes = y_pred.shape[1]

        y_true_dummy = torch.eye(num_classes)[y_true.squeeze(1)]
        y_true_dummy = y_true_dummy.permute(0, 3, 1, 2).to(y_true.device)

        y_pred_proba = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_pred_proba * y_true_dummy, dim=(2, 3))
        dice_loss = ((2 * intersection + self.__EPSILON) / (
                torch.sum(y_pred_proba ** 2 + y_true_dummy ** 2, dim=(2, 3)) + self.__EPSILON))

        return 1 - dice_loss.mean()


class IoULoss(nn.Module):
    """
    Implementation of mean soft-IoU loss for semantic segmentation
    """
    __EPSILON = 1e-6

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
        y_pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        y_true: a tensor of shape [B, H, W].
        Returns:
        float: soft-iou loss.
        """

        # Convert 3d masks to 2d
        if len(y_pred.shape) == 5:
            y_pred = y_pred.view(*y_pred.shape[:-2], -1)
            y_true = y_true.view(*y_true.shape[:-2], -1)

        num_classes = y_pred.shape[1]

        y_true_dummy = torch.eye(num_classes)[y_true.squeeze(1)]
        y_true_dummy = y_true_dummy.permute(0, 3, 1, 2).to(y_true.device)

        y_pred_proba = F.softmax(y_pred, dim=1)

        intersection = torch.sum(y_pred_proba * y_true_dummy, dim=(2, 3))
        union = torch.sum(y_pred_proba ** 2 + y_true_dummy ** 2, dim=(2, 3)) - intersection
        iou_loss = ((intersection + self.__EPSILON) / (union + self.__EPSILON))

        return - iou_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma, weight=None, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"Reduction type must be one of following: 'none' | 'mean' | 'sum'")
        self.reduction = reduction

    def forward(self, inputs, targets):
        cce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        focal_loss = (1 - torch.exp(-cce_loss)) ** self.gamma * cce_loss
        if self.weight is not None:
            focal_loss = self.weight[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean() if self.weight is None else focal_loss.sum() / self.weight[targets].sum()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ParametricKLDivergence(nn.Module):
    def forward(self, dist_params):
        mu, sigma, numel = dist_params
        batch_size = mu.size(0)
        return torch.sum(mu ** 2 + sigma ** 2 - torch.log(sigma ** 2) - 1) / numel / batch_size



def get_loss(loss_config,device):
    loss_name = loss_config['name']
    if loss_name == 'categorical_cross_entropy':
        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'msssim':
        return MSSSIM()
    elif loss_name == 'focal_loss':
        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
        return FocalLoss(gamma=loss_config['gamma'], weight=class_weights)
    elif loss_name == 'mean_iou':
        return IoULoss()
    elif loss_name == 'mean_dice':
        return DiceLoss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'parametric_kl':
        return ParametricKLDivergence()
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")
