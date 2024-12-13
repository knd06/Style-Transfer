import torch
import torch.nn.functional as F

def content_loss(src_feats, dst_feats):
    loss = sum(F.mse_loss(src, dst) for src, dst in zip(src_feats, dst_feats))
    return loss / len(src_feats)

def gram(x):
    N, C, H, W = x.shape
    x = x.view(N * C, -1)  # flatten spatial dimensions
    return torch.mm(x, x.t()) / (H * W)

def style_loss(src_feats, dst_feats, weights):
    loss = 0
    for src, dst, weight in zip(src_feats, dst_feats, weights):
        gram_src = gram(src)
        gram_dst = gram(dst)
        loss += weight * F.mse_loss(gram_src, gram_dst)
    return loss / len(src_feats)