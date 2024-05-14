import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6, multiclass=True):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    dim_check = (2, 3)[multiclass]
    if input.dim() == dim_check and reduce_batch_first:
        raise ValueError(f'Asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == dim_check or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(input, target, reduce_batch_first=False, multiclass=multiclass)


def acc_sens_spec(input: Tensor, target: Tensor, reduce_batch_first: bool = False, multiclass=True):
    assert input.size() == target.size()
    dim_check = (2, 3)[multiclass]
    if input.dim() == dim_check and reduce_batch_first:
        raise ValueError(f'Asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == dim_check or reduce_batch_first:
        inter = torch.logical_and(input, target)
        if multiclass:
            acc = torch.sum(torch.any(inter, dim=0)) / torch.sum(torch.any(target, dim=0))
        else:
            acc = torch.sum(inter) / torch.sum(target)
        return acc
    else:
        # compute and average metric for each batch element
        acc = 0
        for i in range(input.shape[0]):
            acc += acc_sens_spec(input[i, ...], target[i, ...])
        return acc / input.shape[0]
