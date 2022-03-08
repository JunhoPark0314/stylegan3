# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from dnnlib.util import EasyDict

from .dataset import MultiResDataLoader
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

class BaseTrainer:
    def __init__(
        self,
        rank,
        cfg,
        ema_rampup=0.05,
        ada_kimg=500,
        image_snapshot_ticks=50,
        network_snapshot_ticks=50,
        total_kimg=25000,
        G_reg_interval=None,
        D_reg_interval=16,
        kimg_per_tick=4,
        ada_interval=4,
        ada_target=None,
        progress_fn=None,
        abort_fn=None,
    ):
        self.cfg = cfg
        # Initialize.
        self.device = torch.device('cuda', rank)
        self.seed = cfg.random_seed * cfg.num_gpus + rank
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark    # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
        torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
        conv2d_gradfix.enabled = True                       # Improves training speed.
        grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

        # Initialize with given parameters
        self.rank = rank
        self.ema_rampup = ema_rampup
        self.ada_kimg = ada_kimg
        self.G_reg_interval = G_reg_interval
        self.D_reg_interval = D_reg_interval
        self.kimg_per_tick = kimg_per_tick
        self.ada_interval = ada_interval
        self.ada_target = ada_target
        self.progress_fn = progress_fn
        self.abort_fn = abort_fn

        # Load training set.
        if rank == 0:
            print('Loading training set...')

        self.dataloader = MultiResDataLoader(
            c = cfg.data,
            rank = rank,
            num_gpus = cfg.num_gpus,
            random_seed = cfg.random_seed,
            batch_size = cfg.batch_size,
            use_labels = cfg.use_labels,
        )
        
        if rank == 0:
            print()
            print('Num images: ', len(self.dataloader.cur_trainset))
            print('Image shape:', self.dataloader.cur_trainset.image_shape)
            print('Label shape:', self.dataloader.cur_trainset.label_shape)
            print()

        batch_size, batch_gpu, ema_kimg = self.dataloader.get_hyper_params()

        # Construct networks.
        if rank == 0:
            print('Constructing networks...')

        common_kwargs = self.get_network_kwargs()
        self.G = dnnlib.util.construct_class_by_name(**cfg.G_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device) # subclass of torch.nn.Module
        self.D = dnnlib.util.construct_class_by_name(**cfg.D_kwargs, **common_kwargs).train().requires_grad_(False).to(self.device) # subclass of torch.nn.Module
        self.G_ema = copy.deepcopy(self.G).eval()

        self.augment_pipe = None
        self.ada_stats = None

        augment_p = cfg.data.augment_p
        augment_kwargs = cfg.data.augment_kwargs

        # Setup augmentation.
        if rank == 0:
            print('Setting up augmentation...')

        if (augment_kwargs is not None) and (augment_p > 0 or self.ada_target is not None):
            self.augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(self.device) # subclass of torch.nn.Module
            self.augment_pipe.p.copy_(torch.as_tensor(augment_p))
            if self.ada_target is not None:
                self.ada_stats = training_stats.Collector(regex='Loss/signs/real')

        self.pg_info  = EasyDict(
            cur_tick=0,
            cur_nimg=0,
            batch_idx=0,
            tick_start_nimg=0,
            done=False,
            num_gpus=cfg.num_gpus,
            device=self.device,
            rank=rank,
            cur_res = self.dataloader.cur_res
        )

        # Resume from existing pickle.
        if (cfg.resume_pkl is not None) and (rank == 0):
            print(f'Resuming from "{cfg.resume_pkl}"')
            with dnnlib.util.open_url(cfg.resume_pkl) as f:
                resume_data = legacy.load_network_pkl(f)
            for name, module in [('G', self.G), ('D', self.D), ('G_ema', self.G_ema), ('augment_pipe', self.augment_pipe)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

            self.pg_info.update(resume_data['progress_info']) 
            self.dataloader.cur_res = self.pg_info.cur_res

        # Print network summary tables.
        if rank == 0:
            z = torch.empty([batch_gpu, self.G.z_dim], device=self.device)
            c = torch.empty([batch_gpu, self.G.c_dim], device=self.device)
            img = misc.print_module_summary(self.G, [z, c])
            misc.print_module_summary(self.D, [img, c])

        # Distribute across GPUs.
        if rank == 0:
            print(f'Distributing across {cfg.num_gpus} GPUs...')
        for module in [self.G, self.D, self.G_ema, self.augment_pipe]:
            if module is not None and cfg.num_gpus > 1:
                for param in misc.params_and_buffers(module):
                    torch.distributed.broadcast(param, src=0)

        # Setup training phases.
        if rank == 0:
            print('Setting up training phases...')
        self.loss = dnnlib.util.construct_class_by_name(device=self.device, G=self.G, D=self.D, augment_pipe=self.augment_pipe, **cfg.loss_kwargs) # subclass of training.loss.Loss

        # Init phase informations. 
        self.init_phase(cfg)

        # Export sample images.
        self.snap_res = None
        if rank == 0:
            print('Exporting sample images...')
            grid_size, grid_z, grid_c, images = self.setup_snapshot_image_grid()
            save_image_grid(images, os.path.join(cfg.run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
            images = torch.cat([self.G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(cfg.run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

        # Initialize logs.
        if rank == 0:
            print('Initializing logs...')

        self.logger = Logger(
            phases=self.phases, 
            run_dir=cfg.run_dir, 
            rank=rank,
            metrics=cfg.metrics)
        self.checkpointer = Checkpointer(
            run_dir=cfg.run_dir,
            G_ema=self.G_ema,
            image_snapshot_ticks=image_snapshot_ticks,
            network_snapshot_ticks=network_snapshot_ticks
        )
        self.total_kimg = total_kimg
    
    def get_network_kwargs(self):
        return dict(c_dim=self.dataloader.cur_trainset.label_dim, 
                    img_resolution=self.dataloader.resolution[-1], 
                    img_channels=self.dataloader.cur_trainset.num_channels)

    @classmethod
    def get_params(self, module):
        return module.parameters()
    
    def init_phase(self, cfg):
        phases = []
        for idx, (name, module, opt_kwargs, reg_interval) in enumerate([('G', self.G, cfg.G_opt_kwargs, self.G_reg_interval), 
                                                    ('D', self.D, cfg.D_opt_kwargs, self.D_reg_interval)]):
            if reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=self.get_params(module), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(self.get_params(module), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
            
            if hasattr(self, 'phases'):
                phases[-1]['opt'].load_state_dict(self.phases[idx]['opt'].state_dict())

        for phase in phases:
            phase.start_event = None
            phase.end_event = None
            if self.rank == 0:
                phase.start_event = torch.cuda.Event(enable_timing=True)
                phase.end_event = torch.cuda.Event(enable_timing=True)
        
        self.phases = phases
    
    def set_resolution(self, res):
        # Change parameter set of model for given resolution
        self.G.set_resolution(res)
        self.D.set_resolution(res)
        self.G_ema.set_resolution(res)
        self.dataloader.set_resolution(res)

        # Re-initialize phases information based on resolution
        self.init_phase(self.cfg)

    def train(self,):
        # Train.
        if self.pg_info.rank == 0:
            print(f'Training for {self.total_kimg} kimg...')
            print()

        self.logger.init()


        while True:
            # Loop 1 tick
            self.loop_one_batch(self.pg_info)
            self.update_per_batch(self.pg_info)

            if (not self.pg_info.done) and (self.pg_info.cur_tick != 0) and \
                (self.pg_info.cur_nimg < self.pg_info.tick_start_nimg + self.kimg_per_tick * 1000):
                continue

            self.update_per_tick(self.pg_info)
            if self.pg_info.done:
                break

        # Done.
        if self.pg_info.rank == 0:
            print()
            print('Exiting...')

    def setup_snapshot_image_grid(self):
        if (self.snap_res != None) and (self.snap_res != self.dataloader.cur_res):
            self.snap_res = self.dataloader.cur_res
            batch_size, batch_gpu, ema_kimg = self.dataloader.get_hyper_params()
            gw = np.clip(1080 // self.dataloader.cur_trainset.image_shape[1], 4, 8)
            gh = np.clip(1920 // self.dataloader.cur_trainset.image_shape[2], 7, 14)
            self.grid_size = (gw, gh)
            self.grid_z = torch.cat(self.grid_z)[:gw*gh].split(batch_gpu)
            self.grid_c = torch.cat(self.grid_c)[:gw*gh].split(batch_gpu)
        elif self.snap_res == None:
            # Real initialize
            self.snap_res = self.dataloader.cur_res
            batch_size, batch_gpu, ema_kimg = self.dataloader.get_hyper_params()
            training_set = self.dataloader.cur_trainset

            rnd = np.random.RandomState(self.seed)
            gw = np.clip(1080 // training_set.image_shape[1], 4, 8)
            gh = np.clip(1920 // training_set.image_shape[2], 7, 14)

            # No labels => show random subset of training samples.
            if not training_set.has_labels:
                all_indices = list(range(len(training_set)))
                rnd.shuffle(all_indices)
                grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
            else:
                # Group training samples by label.
                label_groups = {} # label => [idx, ...]
                for idx in range(len(training_set)):
                    label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
                    if label not in label_groups:
                        label_groups[label] = []
                    label_groups[label].append(idx)

                # Reorder.
                label_order = sorted(label_groups.keys())
                for label in label_order:
                    rnd.shuffle(label_groups[label])

                # Organize into grid.
                grid_indices = []
                for y in range(gh):
                    label = label_order[y % len(label_order)]
                    indices = label_groups[label]
                    grid_indices += [indices[x % len(indices)] for x in range(gw)]
                    label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

            # Load data.
            images, labels = zip(*[training_set[i] for i in grid_indices])
            self.images = np.stack(images)
            labels = np.stack(labels)
            self.grid_z = torch.randn([labels.shape[0], self.G.z_dim], device=self.device).split(batch_gpu)
            self.grid_c = torch.from_numpy(labels).to(self.device).split(batch_gpu)
            self.grid_size = (gw, gh)

        return self.grid_size, self.grid_z, self.grid_c, self.images

    def update_per_tick(
        self,
        progress_info,
    ):
        """
        progress_info (type : EasyDict)
            - cur_tick          : Current tick
            - cur_nimg          : Current number of image trained
            - batch_idx         : Current batch index
            - tick_start_nimg   : Number of image trained when tick started
            - done              : Flag for stop training
            - num_gpus          : Number of GPUs when multi gpu training
            - device            : Current device id used in training
        """
        # Print status line, accumulating the same information in training_stats.

        self.logger.print_log(progress_info, self.augment_pipe)

        # Check for abort.
        if (not progress_info.done) and (self.abort_fn is not None) and self.abort_fn():
            progress_info.done = True
            if progress_info.rank == 0:
                print()
                print('Aborting...')

        grid_size, grid_z, grid_c, _ = self.setup_snapshot_image_grid()
        self.checkpointer.save_image(progress_info, grid_z, grid_c, grid_size)
        snap_progress = EasyDict(
            cur_tick=self.pg_info.cur_tick,
            cur_nimg=self.pg_info.cur_nimg,
            batch_idx=self.pg_info.batch_idx,
            tick_start_nimg=self.pg_info.tick_start_nimg,
            cur_res=self.pg_info.cur_res,
        )
        snapshot_data = dict(G=self.G, D=self.D, G_ema=self.G_ema, augment_pipe=self.augment_pipe, #training_set_kwargs=dict(dataloader.cur_trainset_kwargs))
                        progress_info=snap_progress)
        snapshot_pkl, snapshot_data = self.checkpointer.save_pkl(progress_info, snapshot_data)

        self.logger.update_metric(snapshot_pkl, snapshot_data, progress_info)

        if snapshot_data:
            del snapshot_data # conserve memory

        self.logger.write_log(progress_info)

        if self.progress_fn is not None:
            self.progress_fn(progress_info.cur_nimg // 1000, self.total_kimg)

        # Update state.
        progress_info.cur_tick += 1
        progress_info.tick_start_nimg = progress_info.cur_nimg
        self.logger.timer.update_start()

    def update_per_batch(
        self,
        progress_info,
    ):
        """
        progress_info (type : EasyDict)
            - cur_tick          : Current tick
            - cur_nimg          : Current number of image trained
            - batch_idx         : Current batch index
            - tick_start_nimg   : Number of image trained when tick started
            - done              : Flag for stop training
            - num_gpus          : Number of GPUs when multi gpu training
            - device            : Current device id used in training
        """
        batch_size, batch_gpu, ema_kimg = self.dataloader.get_hyper_params()
        G = self.G
        G_ema = self.G_ema

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if self.ema_rampup is not None:
                ema_nimg = min(ema_nimg, progress_info.cur_nimg * self.ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        progress_info.cur_nimg += batch_size
        progress_info.batch_idx += 1

        # Execute ADA heuristic.
        if (self.ada_stats is not None) and (progress_info.batch_idx % self.ada_interval == 0):
            self.ada_stats.update()
            adjust = np.sign(self.ada_stats['Loss/signs/real'] - self.ada_target) * (batch_size * self.ada_interval) / (self.ada_kimg * 1000)
            self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(misc.constant(0, device=progress_info.device)))

        # Perform maintenance tasks once per tick.
        progress_info.done = (progress_info.cur_nimg >= self.total_kimg * 1000)

    def loop_one_batch(
        self,
        progress_info,
    ):
        # sourcery skip: simplify-len-comparison, use-fstring-for-concatenation, use-named-expression
        batch_size, batch_gpu, ema_kimg = self.dataloader.get_hyper_params()
        z_dim = self.G.z_dim
        training_set = self.dataloader.cur_trainset
        training_set_iterator = self.dataloader.cur_iterator
        loss = self.loss
        phases = self.phases

        device = progress_info.device
        batch_idx = progress_info.batch_idx
        num_gpus = progress_info.num_gpus
        cur_nimg = progress_info.cur_nimg

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

class ProgressiveTrainer(BaseTrainer):
    def __init__(self, *args, alpha_kimg, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_kimg = alpha_kimg
        self.alpha_idx = 0
        self.max_alpha_idx = int(np.log2(self.dataloader.resolution[-1] / self.dataloader.resolution[0]))

    def get_network_kwargs(self):
        return dict(c_dim=self.dataloader.cur_trainset.label_dim, 
                    target_resolutions=self.dataloader.resolution, 
                    img_channels=self.dataloader.cur_trainset.num_channels)
    
    def update_per_tick(self, progress_info):
        super().update_per_tick(progress_info)

        # change resolution if current nimg is over alpha_kimg
        if (progress_info.cur_nimg + self.alpha_kimg) // (2 * self.alpha_kimg) > self.alpha_idx and self.alpha_idx < self.max_alpha_idx:
            self.alpha_idx = (progress_info.cur_nimg + self.alpha_kimg) // (2 * self.alpha_kimg)
            self.set_resolution(self.dataloader.cur_res * 2)

            if progress_info.rank == 0:
                print(f'Increase target resolution from {self.dataloader.cur_res //2 } to {self.dataloader.cur_res}')
                batch_size, batch_gpu, ema_kimg = self.dataloader.get_hyper_params()
                print(f'Batch Size : {batch_size}, Batch per GPU: {batch_gpu}, ema_kimg: {ema_kimg}')

    @classmethod
    def get_params(self, module):
        return module.parameters()

    def update_per_batch(
        self,
        progress_info,
    ):
        super().update_per_batch(progress_info)
        batch_size, batch_gpu, ema_kimg = self.dataloader.get_hyper_params()
        cur_alpha = torch.ones([]) * (batch_size / self.alpha_kimg)
        self.G.synthesis.alpha.copy_(torch.clip(self.G.synthesis.alpha + cur_alpha, min=0, max=1))
        self.D.alpha.copy_(torch.clip(self.D.alpha + cur_alpha, min=0, max=1))

class Logger:
    def __init__(
        self, 
        phases,
        run_dir,
        rank,
        metrics,
    ):
        self.timer = Timer()
        self.phases = phases
        self.run_dir = run_dir
        self.stats_collector = training_stats.Collector(regex='.*')
        self.stats_metrics = {}
        self.stats_jsonl = None
        self.stats_tfevents = None
        self.metrics = metrics
        if rank == 0:
            self.stats_jsonl = open(os.path.join(self.run_dir, 'stats.jsonl'), 'wt')
            try:
                import torch.utils.tensorboard as tensorboard
                self.stats_tfevents = tensorboard.SummaryWriter(self.run_dir)
            except ImportError as err:
                print('Skipping tfevents export:', err)
    
    def init(self,):
        self.timer.init()

    def print_log(
        self,
        progress_info,
        augment_pipe
    ):
        cur_tick = progress_info.cur_tick
        cur_nimg = progress_info.cur_nimg
        tick_start_nimg = progress_info.tick_start_nimg
        rank = progress_info.rank
        device = progress_info.device
        
        maintenance_time = self.timer.maintenance_time()
        self.timer.update_end()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', self.timer.total_time())):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', self.timer.tick_time()):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', self.timer.tick_time() / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', self.timer.total_time() / (60 * 60))
        training_stats.report0('Timing/total_days', self.timer.total_time() / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))
        
    def update_metric(
        self,
        snapshot_pkl,
        snapshot_data,
        progress_info
    ):
        rank = progress_info.rank
        num_gpus = progress_info.num_gpus
        device = progress_info.device
        
        if (snapshot_data is not None) and (len(self.metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in self.metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=self.dataloader.cur_trainset_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=self.run_dir, snapshot_pkl=snapshot_pkl)
                self.stats_metrics.update(result_dict.results)
    
    def write_log(self, progress_info):
        cur_nimg = progress_info.cur_nimg
        # Collect statistics.
        for phase in self.phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        self.stats_collector.update()
        stats_dict = self.stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if self.stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            self.stats_jsonl.write(json.dumps(fields) + '\n')
            self.stats_jsonl.flush()

        if self.stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = self.timer.wall_time()
            for name, value in stats_dict.items():
                self.stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in self.stats_metrics.items():
                self.stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            self.stats_tfevents.flush()

class Timer:
    def __init__(
        self,
    ):
        self.init()
    
    def init(self):
        self.start_time = time.time()
        self.tick_start_time = time.time()
        self.tick_end_time = time.time()
    
    def update_start(self,):
        self.tick_start_time = time.time()
    
    def update_end(self,):
        self.tick_end_time = time.time()

    def maintenance_time(self,):
        return self.tick_start_time - self.tick_end_time
    
    def total_time(self,):
        return self.tick_end_time - self.start_time

    def tick_time(self,):
        return self.tick_end_time - self.tick_start_time
    
    def wall_time(self,):
        return time.time() - self.start_time

class Checkpointer:
    def __init__(
        self,
        run_dir,
        G_ema,
        image_snapshot_ticks,
        network_snapshot_ticks,
    ):
        self.image_snapshot_ticks = image_snapshot_ticks
        self.network_snapshot_ticks = network_snapshot_ticks
        self.run_dir = run_dir
        self.G_ema = G_ema

    def save_image(
        self, 
        progress_info,
        grid_z,
        grid_c,
        grid_size,
    ):
        done = progress_info.done
        cur_nimg = progress_info.cur_nimg
        cur_tick = progress_info.cur_tick
        rank = progress_info.rank

        # Save image snapshot.
        if (rank == 0) and (self.image_snapshot_ticks is not None) and (done or cur_tick % self.image_snapshot_ticks == 0):
            images = torch.cat([self.G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(self.run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

    def save_pkl(
        self, 
        progress_info,
        snapshot_data
    ):
        done = progress_info.done
        cur_nimg = progress_info.cur_nimg
        cur_tick = progress_info.cur_tick
        num_gpus = progress_info.num_gpus
        rank = progress_info.rank

        # Save network snapshot.
        snapshot_pkl = None
        if (self.network_snapshot_ticks is not None) and (done or cur_tick % self.network_snapshot_ticks == 0):
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value # conserve memory
            snapshot_pkl = os.path.join(self.run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
        else:
            snapshot_data=None
        
        return snapshot_pkl, snapshot_data