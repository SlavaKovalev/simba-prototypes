import torch


def relative_error(computed, expected, eps=1e-38):
    return torch.abs(
        (computed - expected) / (computed + eps)
    ).mean()
