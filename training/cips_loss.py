# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class EBGANLoss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.style_mixing_prob  = style_mixing_prob

    def run_G(self, z, c, coord, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, coord, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, coord, update_emas=False):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        recon_loss = self.D(img, c, coord, update_emas=update_emas)
        return recon_loss

    def accumulate_gradients(self, phase, real_img, real_c, real_coord, gen_z, gen_c, gen_coord, gain, cur_nimg):
        assert phase in ['Gmain', 'Dmain']

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_patches, _gen_ws = self.run_G(gen_z, gen_c, gen_coord)
                loss_Gmain = self.run_D(gen_patches, gen_c, gen_coord)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        if phase in ['Dmain']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_patches, _gen_ws = self.run_G(gen_z, gen_c, gen_coord, update_emas=True)
                loss_Dgen = self.run_D(gen_patches, gen_c, gen_coord, update_emas=True)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

            with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach()
                loss_Dreal = self.run_D(real_img_tmp, real_c, real_coord)
            with torch.autograd.profiler.record_function('Dreal_backward'):
                loss_Dreal.mean().mul(gain).backward()

#----------------------------------------------------------------------------