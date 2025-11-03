from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from ..base_model import BaseModel
from torch.nn import functional as F
# from .baseline import Baseline2
from .BigGait_utils.BigGait_GaitBase import Baseline1
from .BigGait_utils.DINOv2 import vit_small
from ..modules import BasicConv2d, SetBlockWrapper
from .EDINOv2 import InsertEventPrompt, padding_resize, pca

class infoDistillation(nn.Module):
    def __init__(self, source_dim, target_dim, p, softmax, Relu, Up=True):
        super(infoDistillation, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.bn_s = nn.BatchNorm1d(source_dim, affine=False)
        self.bn_t = nn.BatchNorm1d(target_dim, affine=False)
        if Relu:
            self.down_sampling = nn.Sequential(
                nn.Linear(source_dim, source_dim//2),
                nn.BatchNorm1d(source_dim//2, affine=False),
                nn.GELU(),
                nn.Linear(source_dim//2, target_dim),
                )
            if Up:
                self.up_sampling = nn.Sequential(
                    nn.Linear(target_dim, source_dim//2),
                    nn.BatchNorm1d(source_dim//2, affine=False),
                    nn.GELU(),
                    nn.Linear(source_dim//2, source_dim),
                    )
        else:
            self.down_sampling = nn.Linear(source_dim, target_dim)
            if Up:
                self.up_sampling = nn.Linear(target_dim, source_dim)
        self.softmax = softmax
        self.mse = nn.MSELoss()
        self.Up = Up

    def forward(self, x):
        # [n, c]
        d_x = self.down_sampling(self.bn_s(self.dropout(x)))
        if self.softmax:
            d_x = F.softmax(d_x, dim=1)
            if self.Up:
                u_x = self.up_sampling(d_x)
                return d_x, torch.mean(self.mse(u_x, x))
            else:
                return d_x, None
        else:
            if self.Up:
                u_x = self.up_sampling(d_x)
                return torch.sigmoid(self.bn_t(d_x)), torch.mean(self.mse(u_x, x))
            else:
                return torch.sigmoid(self.bn_t(d_x)), None

class Edinov2(nn.Module):
    def __init__(self, model_cfg, logger):
        super().__init__()
        # set input size
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        p = model_cfg['in_channel']
        # set feature dim
        self.f4_dim = 384
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
        self.event_backbone = vit_small(logger=logger)

    def forward(self, x, n, s):
        ns,c,h,w = x.shape
        k = s
        x_mid4 = []
        # idx_mid4 = [2,5,8,11]
        idx_mid4 = [int(i * len(self.event_backbone.blocks) / 4 + len(self.event_backbone.blocks) / 4 - 1) for i in
                    range(4)]
        assert len(idx_mid4) == 4
        c_outs = self.conv1(x)
        x = self.event_backbone.prepare_tokens_with_masks(c_outs)  # ns,hw,d
        for i, blk in enumerate(self.event_backbone.blocks):
            x = self.InsertEventPrompt[i](x, n, k)
            x = blk(x)
            if self.InsertEventPrompt[i].use_event_modality_prompts:
                x = torch.cat((x[:, :1, :], x[:, self.InsertEventPrompt[i].num_event_modality_prompts + 1:, :]),dim=1)
            if i in idx_mid4:
                x_mid4.append(x)
        x_mid4 = partial(nn.LayerNorm, eps=1e-6)(x_mid4[0].shape[-1] * 4, elementwise_affine=False)(
            torch.concat(x_mid4, dim=-1))
        final = self.event_backbone.norm(x)
        c_f_last1 = final[:, 1:].contiguous()
        c_f_last4 = x_mid4[:, 1:].contiguous()

        return c_f_last1, c_f_last4

class EDINOv2_Gait(BaseModel):
    def build_network(self, model_cfg):
        # get pretained models
        self.pretrained_edinov2 = model_cfg["pretrained_edinov2"]
        self.edinov2 = Edinov2(model_cfg, logger=self.msg_mgr)
        self.gait_net = Baseline1(model_cfg)
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.Appearance_Branch = infoDistillation(**model_cfg["Appearance_Branch"])
        self.Action_Branch = infoDistillation(**model_cfg["Action_Branch"])
        self.f4_dim = 384
        self.fc_dim = self.f4_dim * 4
        self.app_dim = model_cfg["Appearance_Branch"]['target_dim']
        self.at_dim = model_cfg["Action_Branch"]['target_dim']
        self.as_dim = model_cfg["Action_Branch"]['source_dim']

    def init_EDINOv2(self):
        self.msg_mgr.log_info(f'load Event_DINOv2 model from: {self.pretrained_edinov2}')
        checkpoint = torch.load(self.pretrained_edinov2)
        model_state_dict = checkpoint['model']
        edinov2_msg = self.edinov2.load_state_dict(model_state_dict, strict=False)
        n_parameters = sum(p.numel() for p in self.edinov2.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(edinov2_msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(edinov2_msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{self.pretrained_edinov2}'")
        self.msg_mgr.log_info('DINOv2 Count: {:.5f}M'.format(n_parameters / 1e6))
        self.edinov2.eval()
        self.edinov2.requires_grad_(False)

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

        self.init_EDINOv2()
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.msg_mgr.log_info('Expect backbone count to train: {:.5f}M'.format(n_parameters / 1e6))

    def preprocess(self, sils, image_size, mode='bilinear'):
        # shape: [nxs,c,h,w] / [nxs,c,224,112]
        return F.interpolate(sils, (image_size * 2, image_size), mode=mode, align_corners=False)

    def min_max_norm(self, x):
        return (x - x.min())/(x.max() - x.min())

    def forward(self, inputs):
        ipts, labs, ty, vi, seqL = inputs
        ratios = None
        with torch.no_grad():  # input_images;         shape: [n,s,c,h,w];
            for c in ipts:
                if len(c.shape) != 5:
                    bb = c
                    ratios = bb[:, :, -2] / bb[:, :, -1]
                else:
                    cef = c
            # real_image_ratios     shape: [n,s,ratio];     ratio: w/h,  e.g. 112/224=0.5;
            del ipts
            n, s, c, h, w = cef.size()
            # sils = rearrange(sils, 'n s c h w -> (n s) c h w').contiguous()
            cef = rearrange(cef, 'n s c h w -> (n s) c h w').contiguous()
            if h == 2 * w:
                # s_outs = self.preprocess(sils, 32)
                c_outs1 = self.preprocess(cef, self.image_size)  # [ns,c,448,224]    if have used pad_resize for input images
            else:
                # s_outs = self.preprocess(padding_resize(sils, ratios, 256, 128), 32)
                c_outs1 = self.preprocess(padding_resize(cef, ratios, 256, 128), self.image_size)  # [ns,c,448,224]    if have not used pad_resize for input images

            c1, c4 = self.edinov2(c_outs1, n, s)
        # c1 = rearrange(c1.view(n, s, self.image_size // 7, self.image_size // 14, -1),
        #                       'n s h w c -> (n s) c h w').contiguous()
        c4 = rearrange(c4.view(n, s, self.image_size // 7, self.image_size // 14, -1),
                              'n s h w c -> (n s) c h w').contiguous()
        outs_c4 = self.preprocess(c4, self.sils_size)
        outs_cef = self.preprocess(cef, self.sils_size)
        outs_c4 = rearrange(outs_c4.view(n, s, -1, self.sils_size * 2, self.sils_size),
                               'n s c h w -> (n s) (h w) c').contiguous()
        outs_cef = rearrange(outs_cef.view(n, s, -1, self.sils_size * 2, self.sils_size),
                               'n s c h w -> (n s) (h w) c').contiguous()
        appearance = outs_c4.view(-1, self.fc_dim)
        app_feat, _ = self.Appearance_Branch(appearance)
        appearance = app_feat.view(n * s, -1, self.app_dim)

        action = outs_cef.view(-1, self.as_dim)
        act_feat, _ = self.Action_Branch(action)
        action = act_feat.view(n * s, -1, self.at_dim)
        embed_1, logits = self.gait_net(
                                        appearance.view(n,s,self.sils_size*2,self.sils_size,self.app_dim).permute(0, 4, 1, 2, 3).contiguous(),
                                        action.view(n,s,self.sils_size*2,self.sils_size,self.at_dim).permute(0, 4, 1, 2, 3).contiguous(),
                                        seqL,
                                        )

        if self.training:
            # vis_c1 = pca(c1)
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embed_1, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {},  #'image/input': s_outs, 'image/c1': torch.from_numpy(vis_c1),
            }
        else:
            retval = {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {'embeddings': embed_1}
            }
        return retval
