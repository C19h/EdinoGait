from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
import math
from operator import mul
from functools import reduce
from kornia import morphology as morph
import random
from collections import OrderedDict
from sklearn.decomposition import PCA
import numpy as np
from .BigGait_utils.BigGait_GaitBase import Baseline
from .BigGait_utils.DINOv2 import vit_small
from .BigGait_utils.save_img import save_image, pca_image
from ..modules import BasicConv2d, SetBlockWrapper
from math import exp

def padding_resize(x, ratios, target_h, target_w):
    n,h,w = x.size(0),target_h, target_w
    ratios = ratios.view(-1)
    need_w = (h * ratios).int()
    need_padding_mask = need_w < w
    pad_left = torch.where(
        need_padding_mask,
        torch.div(w - need_w, 2, rounding_mode='floor'),  
        torch.zeros(1, dtype=need_w.dtype).to(x.device)
    )
    pad_right = torch.where(need_padding_mask, w - need_w - pad_left, torch.zeros(1, dtype=need_w.dtype).to(x.device)).tolist()
    need_w = need_w.tolist()
    pad_left = pad_left.tolist()

    x = torch.concat([F.pad(F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False), (pad_left[i], pad_right[i]))  if need_padding_mask[i] else F.interpolate(x[i:i+1,...], (h, need_w[i]), mode="bilinear", align_corners=False)[...,pad_left[i]:pad_left[i]+w]  for i in range(n)], dim=0)
    return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
            self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
            qk_proj_dim: int, v_proj_dim: int, num_heads: int,
            out_dim: int
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0);
        assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1);
        assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out

class InsertEventPrompt(nn.Module):
    def __init__(self, cfg, patch_size, feature_dim, num_heads):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.use_event_modality_prompts = cfg['EventEncoder']['use_event_modality_prompts']
        self.num_event_modality_prompts = cfg['EventEncoder']['num_event_modality_prompts']
        if self.use_event_modality_prompts:
            self.event_modality_prompts = nn.Parameter(torch.zeros(self.num_event_modality_prompts, feature_dim), requires_grad=True)
            self._initialize_event_modality_prompts(self.patch_size, feature_dim)


    def _initialize_event_modality_prompts(self, patch_size, prompt_dim):
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.event_modality_prompts.data, -val, val)

    def forward(self, x, B, T):
        device = x.device
        if self.use_event_modality_prompts:
            event_modality_prompts = self.event_modality_prompts.expand(B * T, -1, -1).to(device)
            # add global_prompts after the cls token while in front of the original token.
            x = torch.cat((x[:, :1, :], event_modality_prompts, x[:, 1:, :]), dim=1)
        return x

class EDINOv2(BaseModel):
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_dinov2 = model_cfg["pretrained_dinov2"]
        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        # set feature dim
        self.f4_dim = 384
        p = model_cfg['in_channel']
        self.conv1 = BasicConv2d(p, 3, 1, 1, 0)
        self.encoder = BasicConv2d(self.f4_dim, p, 1, 1, 0)
        self.decoder = BasicConv2d(p, self.f4_dim, 1, 1, 0)
        n_blocks = 12
        self.InsertEventPrompt = nn.ModuleList()
        for i in range(n_blocks):
            self.InsertEventPrompt.append(
                InsertEventPrompt(model_cfg, patch_size=14, feature_dim=self.f4_dim, num_heads=12))

        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.mse = nn.MSELoss()


    def init_DINOv2(self):
        self.image_backbone = vit_small(logger = self.msg_mgr)
        self.msg_mgr.log_info(f'load image_DINOv2 model from: {self.pretrained_dinov2}')
        pretrain_dict = torch.load(self.pretrained_dinov2)
        image_msg = self.image_backbone.load_state_dict(pretrain_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.image_backbone.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(image_msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(image_msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_dinov2}'")
        self.msg_mgr.log_info('DINOv2 Count: {:.5f}M'.format(n_parameters / 1e6))
        self.image_backbone.eval()
        self.image_backbone.requires_grad_(False)

        self.event_backbone = vit_small(logger = self.msg_mgr)
        self.msg_mgr.log_info(f'load event_DINOv2 model from: {self.pretrained_dinov2}')
        pretrain_dict = torch.load(self.pretrained_dinov2)
        msg = self.event_backbone.load_state_dict(pretrain_dict, strict=True)
        if self.training:
            self.event_backbone.train()
            self.event_backbone.requires_grad_(True)

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        self.init_DINOv2()
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.msg_mgr.log_info('Expect backbone count to train: {:.5f}M'.format(n_parameters / 1e6))

    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size * 2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def symmetric_cross_entropy_loss(self, logits):
        # symmetric loss function
        batch_size = logits.shape[0]
        device = logits.device
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2
        return loss

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        ratios = None
        with torch.no_grad(): # input_images;         shape: [n,s,c,h,w];
            for c in ipts:
                if len(c.shape) != 5:
                    bb = c
                    ratios = bb[:, :, -2] / bb[:, :, -1]
                elif c.shape[2] == 3:
                    sils = c
                else:
                    cef = c
               # real_image_ratios     shape: [n,s,ratio];     ratio: w/h,  e.g. 112/224=0.5;
            del ipts
            n,s,c,h,w = sils.size()
            sils = rearrange(sils, 'n s c h w -> (n s) c h w').contiguous()
            cef = rearrange(cef, 'n s c h w -> (n s) c h w').contiguous()
            if h == 2*w:
                s_outs = self.preprocess(sils, self.image_size)                                           # [ns,c,448,224]    if have used pad_resize for input images
                c_outs1 = self.preprocess(cef, self.image_size)                                           # [ns,c,448,224]    if have used pad_resize for input images
                cef_outs32 = self.preprocess(cef, 16)                                           # [ns,c,448,224]    if have used pad_resize for input images
            else:
                s_outs = self.preprocess(padding_resize(sils, ratios, 256, 128), self.image_size)         # [ns,c,448,224]    if have not used pad_resize for input images
                c_outs1 = self.preprocess(padding_resize(cef, ratios, 256, 128), self.image_size)         # [ns,c,448,224]    if have not used pad_resize for input images
                cef_outs32 = self.preprocess(padding_resize(cef, ratios, 256, 128), 16)         # [ns,c,448,224]    if have not used pad_resize for input images
            s_f = self.image_backbone(s_outs, is_training=True)
        s_f_last1 = s_f["x_norm_patchtokens"].contiguous()
        # vis_f(s_f_last1, 30, './s.png')
        s_f_last4 = s_f["x_norm_patchtokens_mid4"].contiguous()
        s_f_cls = s_f["cls"].contiguous()
        x_mid4 = []
        # idx_mid4 = [2,5,8,11]
        idx_mid4 = [int(i * len(self.event_backbone.blocks) / 4 + len(self.event_backbone.blocks) / 4 - 1) for i in range(4)]
        assert len(idx_mid4) == 4
        c_outs = self.conv1(c_outs1)
        x = self.event_backbone.prepare_tokens_with_masks(c_outs) # ns,hw,d
        for i, blk in enumerate(self.event_backbone.blocks):
            x = self.InsertEventPrompt[i](x, n, s)
            x = blk(x)
            if self.InsertEventPrompt[i].use_event_modality_prompts:
                x = torch.cat((x[:, :1, :], x[:, self.InsertEventPrompt[i].num_event_modality_prompts + 1:, :]), dim=1)
            if i in idx_mid4:
                x_mid4.append(x)

        x_mid4 = partial(nn.LayerNorm, eps=1e-6)(x_mid4[0].shape[-1] * 4, elementwise_affine=False)(torch.concat(x_mid4, dim=-1))
        final = self.event_backbone.norm(x)
        c_f_last1 = final[:, 1:].contiguous()
        # vis_f(c_f_last1, 30, './c.png')
        c_f_last4 = x_mid4[:, 1:].contiguous()
        c_f_cls = final[:, 0].contiguous()

        logits_s_c = c_f_cls @ s_f_cls.t()
        loss_lo_s_c = self.symmetric_cross_entropy_loss(logits_s_c)
        s_f_last1 = rearrange(s_f_last1.view(n, s, self.image_size // 7, self.image_size // 14, -1),
                              'n s h w c -> (n s) c h w').contiguous()
        c_f_last1 = rearrange(c_f_last1.view(n, s, self.image_size // 7, self.image_size // 14, -1),
                              'n s h w c -> (n s) c h w').contiguous()


        vis_s_f_last1 = pca(s_f_last1)
        c = self.encoder(c_f_last1)
        r_c_f_last1 = self.decoder(c)
        vis_r_c_f_last1 = pca(r_c_f_last1)
        vis_c_f_last1 = pca(c_f_last1)
        # loss_ssim = self.ssim_loss(c, cef_outs32)
        loss_mse = self.mse(c_f_last1, r_c_f_last1)
        loss_hid = self.mse(c, cef_outs32)

        # vis_gray(vis_s_f_last1[10].cpu().detach().permute(1,2,0),'./sf.png')
        # vis_gray(vis_c_outs1[10].cpu().detach().permute(1,2,0),'./c_outs.png')
        # c_f_last4 = rearrange(c_f_last4.view(n, s, self.image_size // 7, self.image_size // 14, -1),
        #                       'n s h w c -> (n s) c h w').contiguous()
        # app = self.conv2(c_f_last4)
        # vis_app = pca(app)
        #
        # appl = rearrange(app, 'n c h w -> n (h w) c').contiguous()
        # diversity_loss = self.diversity_loss(appl, 16)

        if self.training:
            retval = {
                'training_feat': {
                    'loss_lo_s_c:': loss_lo_s_c*0.1,
                    'loss_hid:': loss_hid*5,
                    'loss_res_mse:': loss_mse*5,
                    # 'loss_diversity:': diversity_loss,
                },
                'visual_summary': {
                    'image/input': s_outs,
                    'image/res_c_f_last1': torch.from_numpy(vis_r_c_f_last1),
                    'image/c_f_last1': torch.from_numpy(vis_c_f_last1),
                    # 'image/app': self.min_max_norm(torch.from_numpy(vis_app)),
                    'image/s_f_last1': torch.from_numpy(vis_s_f_last1),
                },
            }
        return retval

def vis_gray(image, path):

    plt.imshow(image, cmap='viridis')  # 使用灰度颜色映射
    plt.colorbar()  # 添加颜色条，表示灰度值范围
    plt.title('Grayscale Image')
    plt.savefig(path)
    plt.close()

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

def pca(f):
    f4_numpy = f.detach().cpu().numpy()  # 将张量转换为 numpy 数组
    n, c, h, w = f4_numpy.shape  # n 是批次大小，c 是通道数，h 和 w 是特征图的高宽

    # 将批次中的特征图展开为二维数组，形状为 (n * h * w, c)
    reshaped = f4_numpy.reshape(n, c, h * w).transpose(0, 2, 1).reshape(-1, c)  # (n * h * w, c)

    # 执行 PCA，将 c 维降到 3 维
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped)  # (n * h * w, 3)

    # 将结果重塑回 (n, h, w, 3) 的形状
    pca_images = pca_result.reshape(n, h, w, 3).transpose(0, 3, 1, 2)

    # 对 PCA 结果进行归一化到 [0, 1]
    pca_images = (pca_images - pca_images.min()) / (pca_images.max() - pca_images.min())

    # 返回整个批次的 PCA 结果
    return pca_images


if __name__ =='__main__':

    model = EDINOv2()