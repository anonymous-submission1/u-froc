import torch
from dpipe.torch import weighted_cross_entropy_with_logits, dice_loss_with_logits

from ufroc.batch_iter import SPATIAL_DIMS


# small target segm loss


def focal_tversky_loss_with_logits(logit, target, spatial_dims, beta, gamma):
    proba = torch.sigmoid(logit)
    intersection = torch.sum(proba * target, dim=spatial_dims)
    tp = torch.sum(proba ** 2 * target, dim=spatial_dims)
    fp = torch.sum(proba ** 2 * (1 - target), dim=spatial_dims)
    fn = torch.sum((1 - proba ** 2) * target, dim=spatial_dims)
    tversky_index = intersection / (tp + beta * fn + (1 - beta) * fp + 1)
    loss = (1 - tversky_index) ** gamma
    return loss.mean()


def small_target_segm_loss(logit, target):
    return (weighted_cross_entropy_with_logits(logit, target)
            + focal_tversky_loss_with_logits(logit, target, spatial_dims=SPATIAL_DIMS, beta=0.7, gamma=1))


def combined_loss_with_logits(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
                              alpha=0.5, beta=0.7, adaptive_bce=False):
    return (1 - alpha) * dice_loss_with_logits(logit, target) \
           + alpha * weighted_cross_entropy_with_logits(logit, target, weight=weight, alpha=beta, adaptive=adaptive_bce)
