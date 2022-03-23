from .CfgNode import CfgNode as CN
from metrics import metric_main

#===============================================================================
# Predefined constants
#===============================================================================
CBASE                                   = 32768
CMAX                                    = 512

_C                                      = CN(new_allowed=True)

#===============================================================================
# Cfg about Generator
# 
# 
#===============================================================================

_C.G_kwargs                             = CN(new_allowed=True)
_C.G_kwargs.class_name                  = "training.networks_stylegan2.Generator"
_C.G_kwargs.z_dim                       = 512 # Size of Latent Z
_C.G_kwargs.w_dim                       = 512 # Size of StyleCode W
_C.G_kwargs.channel_base                = CBASE
_C.G_kwargs.channel_max                 = CMAX
_C.G_kwargs.conv_kernel                 = 1

#-------------------------------------------------------------------------------
# Cfg about Generator.MappingNetwork
#-------------------------------------------------------------------------------
_C.G_kwargs.mapping_kwargs              = CN()
_C.G_kwargs.mapping_kwargs.num_layers   = 8

#-------------------------------------------------------------------------------
# Cfg about Generator's optimizer
#-------------------------------------------------------------------------------

_C.G_opt_kwargs                  = CN()
_C.G_opt_kwargs.class_name       = "torch.optim.Adam"
_C.G_opt_kwargs.betas            = [0, 0.99]
_C.G_opt_kwargs.eps              = 1e-8
_C.G_opt_kwargs.lr               = 0.0025

#===============================================================================
# Cfg about Discriminator's configuration
# 
# 
#===============================================================================

_C.D_kwargs                             = CN(new_allowed=True)
_C.D_kwargs.class_name                  = "training.networks_stylegan2.Discriminator"
_C.D_kwargs.channel_base                = CBASE
_C.D_kwargs.channel_max                 = CMAX
_C.D_kwargs.conv_kernel                 = 1

#-------------------------------------------------------------------------------
# Cfg about Discriminator.DiscriminatorBlock
#-------------------------------------------------------------------------------
_C.D_kwargs.block_kwargs                = CN() 
_C.D_kwargs.block_kwargs.freeze_layers  = 0

#-------------------------------------------------------------------------------
# Cfg about Discriminator.MappingNetwork
#-------------------------------------------------------------------------------
_C.D_kwargs.mapping_kwargs              = CN() 

#-------------------------------------------------------------------------------
# Cfg about Discriminator.DiscriminatorEpilogue
#-------------------------------------------------------------------------------
_C.D_kwargs.epilogue_kwargs             = CN() 
_C.D_kwargs.epilogue_kwargs.mbstd_group_size = 4

#-------------------------------------------------------------------------------
# Cfg about Discriminator's optimizer
#-------------------------------------------------------------------------------

_C.D_opt_kwargs                  = CN()
_C.D_opt_kwargs.class_name       = "torch.optim.Adam"
_C.D_opt_kwargs.betas            = [0, 0.99]
_C.D_opt_kwargs.eps              = 1e-8
_C.D_opt_kwargs.lr               = 0.002

#===============================================================================
# Cfg about Loss configuration
# 
# 
#===============================================================================

_C.loss_kwargs                          = CN(new_allowed=True)
_C.loss_kwargs.class_name               = "training.loss.StyleGAN2Loss"
_C.loss_kwargs.r1_gamma                 = 2.0

#===============================================================================
# Cfg about Dataset / DataLoader configuration
# 
# 
#===============================================================================

_C.data                                 = CN()
_C.data.dataset_name                    = "afhq-v2"
_C.data.xflip                           = True
_C.data.size                            = 1000
_C.data.resolution                      = [128,]
_C.data.class_name                      = "training.dataset.ImageFolderDataset"
_C.data.aug                             = "ada"
_C.data.augment_p                       = 0.0
_C.data.max_batch                       = 32


_C.data.loader_kwargs                   = CN()
_C.data.loader_kwargs.pin_memory        = True
_C.data.loader_kwargs.prefetch_factor   = 2
_C.data.loader_kwargs.num_workers       = 3

# arguments for augmentation pipe line
_C.data.augment_kwargs                  = CN()
_C.data.augment_kwargs.class_name       = "training.augment.AugmentPipe"
_C.data.augment_kwargs.xflip            = 1
_C.data.augment_kwargs.rotate90         = 1
_C.data.augment_kwargs.xint             = 1
_C.data.augment_kwargs.scale            = 1
_C.data.augment_kwargs.rotate           = 1
_C.data.augment_kwargs.aniso            = 1
_C.data.augment_kwargs.xfrac            = 1
_C.data.augment_kwargs.brightness       = 1
_C.data.augment_kwargs.contrast         = 1
_C.data.augment_kwargs.lumaflip         = 1
_C.data.augment_kwargs.hue              = 1
_C.data.augment_kwargs.saturation       = 1

#===============================================================================
# Cfg about Training loop configuration
# 
# 
#===============================================================================

_C.loop_kwargs                          = CN(new_allowed=True)
_C.loop_kwargs.class_name               = "training.training_loop.BaseTrainer"
_C.loop_kwargs.ema_rampup               = 0.05
_C.loop_kwargs.ada_kimg                 = 100
_C.loop_kwargs.image_snapshot_ticks     = 20
_C.loop_kwargs.network_snapshot_ticks   = 20
_C.loop_kwargs.total_kimg               = 25000
_C.loop_kwargs.D_reg_interval           = 16
_C.loop_kwargs.kimg_per_tick            = 4
_C.loop_kwargs.ada_target               = 0.6
_C.loop_kwargs.ada_interval             = 4

#===============================================================================
# Cfg about Basic configuration
# 
# 
#===============================================================================

_C.num_gpus                             = 1
_C.batch_size                           = 32
_C.random_seed                          = 0
_C.resume_pkl                           = None
_C.cudnn_benchmark                      = True
_C.outdir                               = None
_C.use_labels                           = False
_C.metrics                              = []


def extract_name(cfg_file):
    path_split = cfg_file.split('/')
    return str.join('_', [path_split[-2], path_split[-1].split('.')[0]])

def parse_opts(opts, cfg_list):
    # Parse opts' option and add to cfg_lists
    opt_cfg_list = ()
    opt_cfg_list += ("num_gpus", opts.gpus)
    opt_cfg_list += ("outdir", opts.outdir)
    opt_cfg_list += ("use_labels", opts.cond)
    opt_cfg_list += ("batch_size", opts.batch)
    opt_cfg_list += ("resume_pkl", opts.resume)
    opt_cfg_list += ("metrics", opts.metrics)

    opt_cfg_list += ("loop_kwargs.total_kimg", opts.kimg)
    opt_cfg_list += ("loop_kwargs.kimg_per_tick", opts.tick)

    opt_cfg_list += ("data.aug", opts.aug)
    opt_cfg_list += ("data.dataset_name", opts.data)
    opt_cfg_list += ("data.xflip", opts.mirror)
    opt_cfg_list += ("data.loader_kwargs.num_workers", opts.workers)

    opt_cfg_list += ("loss_kwargs.r1_gamma", opts.gamma)

    opt_cfg_list += ("random_seed", opts.seed)

    return cfg_list + opt_cfg_list

def get_cfg_defaults():
    return _C

def get_cfg(opts, cfg_list):
    cfg_list = parse_opts(opts, cfg_list)

    # Get default cfgs and merge with target cfg file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opts.cfg_file)

    # Update argument unaware cfgs based on initialized cfg (e.g. FP32, benchmark)
    if opts.snap:
        cfg.loop_kwargs.image_snapshot_ticks = cfg.loop_kwargs.network_snapshot_ticks = opts.snap

    if opts.fp32:
        cfg.G_kwargs.num_fp16_res = cfg.D_kwargs.num_fp16_res = 0
        cfg.G_kwargs.conv_clamp = cfg.D_kwargs.conv_clamp = None

    if opts.nobench:
        cfg.cudnn_benchmark = False
    
    if opts.resolution:
        cfg.data.resolution = opts.resolution
    cfg.data.resolution = sorted(cfg.data.resolution)
    
    cfg.expr_name = extract_name(opts.cfg_file)

    cfg.merge_from_list(cfg_list)
    cfg.clear_build()

    # Update argument aware cfgs based on initialized cfg (e.g. Batch size, )
    cfg.ema_kimg = cfg.batch_size * 10 / 32
    min_batch_size = (int((min(cfg.data.resolution) / max(cfg.data.resolution)) * cfg.batch_size) // cfg.num_gpus) * cfg.num_gpus
    min_batch_gpu = max(min_batch_size // cfg.num_gpus, 4)
    
    # Sanity check for configuration
    if cfg.batch_size % cfg.num_gpus != 0:
        raise Exception('--batch must be a multiple of --gpus')
    if min_batch_gpu < cfg.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise Exception('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in cfg.metrics):
        raise Exception('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
        
    # Description string.
    res_string = ":".join([str(res) for res in cfg.data.resolution])
    desc = f'{cfg.expr_name:s}-{cfg.data.dataset_name:s}-{res_string:s}-gpus{cfg.num_gpus:d}-batch{cfg.batch_size:d}-gamma{cfg.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    return cfg, desc