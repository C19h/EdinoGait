
import os
import argparse
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import config_loader, my_get_msg_mgr
from opengait.modeling.models.BigGait_utils.DINOv2 import vit_small
from opengait.modeling.models.Edinov2_gait import Edinov2

from scipy.sparse import load_npz
import numpy as np
import pickle
from functools import partial
from sklearn.decomposition import PCA
parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='configs/edinov2_reg_cef_hnugait.yaml', help="path of config file")

opt = parser.parse_args()


def preprocess(sils, image_size, mode='bilinear'):
    # shape: [nxs,c,h,w] / [nxs,c,224,112]
    return F.interpolate(sils, (image_size * 2, image_size), mode=mode, align_corners=False)

def padding_resize(x, ratios, target_h, target_w):
    n,h,w = x.size(0),target_h, target_w
    ratios = ratios.view(-1)
    need_w = (h * ratios).int()
    need_padding_mask = need_w < w
    pad_left = torch.where(
        need_padding_mask,
        torch.div(w - need_w, 2, rounding_mode='floor'),  # 使用 `floor` 进行向下取整
        torch.zeros(1, dtype=need_w.dtype).to(x.device)
    )
    pad_right = torch.where(need_padding_mask, w - need_w - pad_left, torch.zeros(1, dtype=need_w.dtype).to(x.device)).tolist()
    need_w = need_w.tolist()
    pad_left = pad_left.tolist()
    x = x.float()
    x = torch.concat([F.pad(F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False), (pad_left[i], pad_right[i]))  if need_padding_mask[i] else F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False)[...,pad_left[i]:pad_left[i]+w]  for i in range(n)], dim=0)
    return x

def vis_gray(image, path):
    plt.imshow(image, cmap='gray')  # 使用灰度颜色映射
    plt.axis('off')
    plt.savefig(path)
    plt.close()

def symmetric_cross_entropy_loss(logits):
    # symmetric loss function
    batch_size = logits.shape[0]
    device = logits.device
    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    loss = (loss_i + loss_t) / 2
    return loss

def vis_f(f, ind, path):
    f4_numpy = f.detach().cpu().numpy()  # 将张量转换为 numpy 数组
    n, c, h, w = f4_numpy.shape  # n 是批次大小，c 是通道数，h 和 w 是特征图的高宽
    # 取出其中一个特征图进行可视化
    feature_map = f4_numpy[ind]  # 选择第一个样本 (384, 32, 16)

    # 将其重塑为 (32*16, 384) 的二维数组，以便进行 PCA
    reshaped = feature_map.reshape(c, -1).T  # (512, 384)
    # 执行 PCA，将 384 维降到 3 维
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped)  # (512, 3)
    # 将结果重塑回 (32, 16, 3)，并归一化到 0-1 之间
    pca_image = pca_result.reshape(h, w, 3)
    pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
    plt.imshow(pca_image)
    plt.axis('off')  # 去掉坐标轴
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
if __name__ == '__main__':
    msg_mgr = my_get_msg_mgr()
    msg_mgr.init_logger('./output/', log_to_file=False)
    cfgs = config_loader(opt.cfgs)
    dinov2 = vit_small(logger=msg_mgr)

    pretrain_dict = torch.load('/root/autodl-tmp/3modality/pretrained/dinov2_vits14_pretrain.pth')
    dinov2.load_state_dict(pretrain_dict, strict=True)
    dinov2.eval()
    dinov2.requires_grad_(False)
    dinov2.to('cuda')
    edinov2 = Edinov2(cfgs['model_cfg'], logger=msg_mgr)
    pretrained_edinov2 = '/root/autodl-tmp/3modality/pretrained/edinov2_cef_hnu.pt'
    checkpoint = torch.load(pretrained_edinov2)
    model_state_dict = checkpoint['model']
    edinov2.load_state_dict(model_state_dict, strict=False)
    edinov2.eval()
    edinov2.requires_grad_(False)
    edinov2.to('cuda')
    data_path = '/root/autodl-tmp/3modality/db/HNU-Gait/light/41/cl-01/045'
    seq_dirs = sorted([os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith('.pkl') or f.endswith('.npz')])
    for pth in seq_dirs:
        if pth.endswith('gray.npz'):
            output = load_npz(pth)
            gray = np.repeat(output.toarray().reshape(-1, 64, 64, 1).transpose(0, 3, 1, 2), repeats=3, axis=1)
            gray = torch.tensor(gray).to('cuda')
        if pth.endswith('continuous.npz'):
            output = load_npz(pth)
            cef = output.toarray().reshape(-1, 64, 64, 8).transpose(0, 3, 1, 2)
            cef = torch.tensor(cef).to('cuda')
        if pth.endswith('bboxes.pkl'):
            with open(pth, 'rb') as f:
                bb = pickle.load(f)['bboxes'].numpy()
                bb = torch.tensor(bb).to('cuda')
                ratios = bb[:, -2] / bb[:, -1]
    s_outs = preprocess(padding_resize(gray, ratios, 256, 128), 224)  # [ns,c,448,224]    if have not used pad_resize for input images
    c_outs1 = preprocess(padding_resize(cef, ratios, 256, 128), 224)
    n = s_outs.shape[0]
    s_f = dinov2(s_outs, is_training=True) # [1,c,32,16]
    s_f_last4 = s_f["x_norm_patchtokens_mid4"].contiguous()
    s_f_last = s_f["x_norm_patchtokens"].contiguous()
    s_f_cls = s_f["cls"].contiguous()
    s_f_last = s_f_last.view(n, 224 // 7, 224 // 14, -1)

    c1, c4 = edinov2(c_outs1, n, 1)
    c1 = c1.view(n, 224 // 7, 224 // 14, -1).contiguous()
    ind = 135
    vis_gray(s_outs.permute(0,2,3,1)[ind][:,:,2].cpu().detach().numpy(), f'./gray_{ind}.png')
    vis_f(s_f_last.permute(0,3,1,2), ind, f'./sf_{ind}.png')
    vis_f(c1.permute(0,3,1,2), ind, f'./c_{ind}.png')
