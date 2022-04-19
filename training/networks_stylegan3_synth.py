# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import bias_act
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torchvision.utils import save_image

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
	def __init__(self,
		in_channels,                    # Number of input channels.
		out_channels,                   # Number of output channels.
		kernel_size,                    # Width and height of the convolution kernel.
		bias            = True,         # Apply additive bias before the activation function?
		activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
		up              = 1,            # Integer upsampling factor.
		down            = 1,            # Integer downsampling factor.
		resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
		channels_last   = False,        # Expect the input to have memory_format=channels_last?
		trainable       = True,         # Update the weights of this layer during training?
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.activation = activation
		self.up = up
		self.down = down
		self.conv_clamp = conv_clamp
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
		self.padding = kernel_size // 2
		self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
		self.act_gain = bias_act.activation_funcs[activation].def_gain

		memory_format = torch.channels_last if channels_last else torch.contiguous_format
		weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
		bias = torch.zeros([out_channels]) if bias else None
		if trainable:
			self.weight = torch.nn.Parameter(weight)
			self.bias = torch.nn.Parameter(bias) if bias is not None else None
		else:
			self.register_buffer('weight', weight)
			if bias is not None:
				self.register_buffer('bias', bias)
			else:
				self.bias = None

	def forward(self, x, gain=1):
		w = self.weight * self.weight_gain
		b = self.bias.to(x.dtype) if self.bias is not None else None
		flip_weight = (self.up == 1) # slightly faster
		x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

		act_gain = self.act_gain * gain
		act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
		x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
		return x

	def extra_repr(self):
		return ' '.join([
			f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
			f'up={self.up}, down={self.down}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
	def __init__(self,
		in_features,                # Number of input features.
		out_features,               # Number of output features.
		activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
		bias            = True,     # Apply additive bias before the activation function?
		lr_multiplier   = 1,        # Learning rate multiplier.
		weight_init     = 1,        # Initial standard deviation of the weight tensor.
		bias_init       = 0,        # Initial value of the additive bias.
	):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.activation = activation
		self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
		bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
		self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
		self.weight_gain = lr_multiplier / np.sqrt(in_features)
		self.bias_gain = lr_multiplier

	def forward(self, x):
		w = self.weight.to(x.dtype) * self.weight_gain
		b = self.bias
		if b is not None:
			b = b.to(x.dtype)
			if self.bias_gain != 1:
				b = b * self.bias_gain
		if self.activation == 'linear' and b is not None:
			x = torch.addmm(b.unsqueeze(0), x, w.t())
		else:
			x = x.matmul(w.t())
			x = bias_act.bias_act(x, b, act=self.activation)
		return x

	def extra_repr(self):
		return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
	def __init__(self,
		z_dim,                      # Input latent (Z) dimensionality.
		c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
		w_dim,                      # Intermediate latent (W) dimensionality.
		num_ws,                     # Number of intermediate latents to output.
		num_layers      = 2,        # Number of mapping layers.
		lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
		w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
	):
		super().__init__()
		self.z_dim = z_dim
		self.c_dim = c_dim
		self.w_dim = w_dim
		self.num_ws = num_ws
		self.num_layers = num_layers
		self.w_avg_beta = w_avg_beta

		# Construct layers.
		self.embed = FullyConnectedLayer(self.c_dim, self.w_dim) if self.c_dim > 0 else None
		features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
		for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
			layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
			setattr(self, f'fc{idx}', layer)
		self.register_buffer('w_avg', torch.zeros([w_dim]))

	def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
		misc.assert_shape(z, [None, self.z_dim])
		if truncation_cutoff is None:
			truncation_cutoff = self.num_ws

		# Embed, normalize, and concatenate inputs.
		x = z.to(torch.float32)
		x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
		if self.c_dim > 0:
			misc.assert_shape(c, [None, self.c_dim])
			y = self.embed(c.to(torch.float32))
			y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
			x = torch.cat([x, y], dim=1) if x is not None else y

		# Execute layers.
		for idx in range(self.num_layers):
			x = getattr(self, f'fc{idx}')(x)

		# Update moving average of W.
		if update_emas:
			self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

		# Broadcast and apply truncation.
		x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
		if truncation_psi != 1:
			x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
		return x

	def extra_repr(self):
		return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------
class SynthesisGroupKernel(torch.nn.Module):
	def __init__(self,
		in_channels,
		out_channels,
		cutoff = None,
		sampling_rate = 16,
		distribution = "sin", # "uniform / biased / data-driven / sin"
		dist_init = None,
		max_freq = 512,
		sort_dist = True,
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.sampling_rate = sampling_rate
		self.cutoff = cutoff * 2 if cutoff is not None else sampling_rate
		self.bandwidth = self.sampling_rate * (2 ** 0.1) #* (2 ** -0.9)
		self.freq_dim = np.clip(self.sampling_rate * 8, a_min=128, a_max=max_freq)

		# Draw random frequencies from uniform 2D disc.
		freqs = torch.randn([self.in_channels, self.freq_dim, 2])
		radii = freqs.square().sum(dim=-1, keepdim=True).sqrt()
		freqs /= radii
		if distribution == "uniform":
			dist = radii.square().exp().pow(-0.25)
		elif distribution == "low_biased":
			dist = torch.rand([self.in_channels, self.freq_dim, 1])
		elif distribution == "high_biased":
			dist = torch.randn([self.in_channels, self.freq_dim, 1]).sin()
		elif distribution == "data-driven":
			assert dist_init is not None
			dist = dist_init
		
		if sort_dist:
			dist = torch.sort(dist, dim=1)[0]

		freqs *= (self.bandwidth * dist)
		phases = torch.rand([self.in_channels, self.freq_dim]) - 0.5

		self.register_buffer("freqs", freqs)
		self.phases = torch.nn.Parameter(phases)

		self.freq_weight = torch.nn.Parameter(torch.randn([in_channels, self.freq_dim]))
		self.freq_out = torch.nn.Parameter(torch.randn([out_channels, self.freq_dim]))
		self.weight_bias = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, 1]))

		self.gain = lambda x: np.sqrt(1 / (in_channels * (x ** 2)))


	def get_freqs(self):
		return self.freqs

	def forward(self, device, ks, alpha=1, update_emas=False, style=None):
		sample_size = ks
		in_freqs = self.get_freqs()
		in_phases = (self.phases).unsqueeze(0)
		in_mag = self.freq_weight.unsqueeze(0) 

		theta = torch.eye(2, 3, device=device)
		theta[0, 0] = 0.5 * sample_size / (self.sampling_rate)
		theta[1, 1] = 0.5 * sample_size / (self.sampling_rate)
		grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, sample_size, sample_size], align_corners=False)

		ix = torch.einsum('bhwr,ifr->bihwf', grids, in_freqs)
		ix = ix + (in_phases).unsqueeze(2).unsqueeze(3)
		ix = torch.sin(ix * (np.pi * 2))
		
		out_mag = self.freq_out
		kernel = torch.einsum('bihwf,bif,of->boihw', ix, in_mag, out_mag) * np.sqrt(1 / (self.freq_dim * 2))
		kernel = kernel + self.weight_bias * np.sqrt(1 / (2 * ks))

		kernel = kernel.squeeze(0) * self.gain(ks)

		return kernel

	def freq_parameters(self):
		params = []
		for k, v in self.named_parameters():
			if k in self.freq_param:
				params.append(v)
		return params

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
	def __init__(self,
		in_channels,                        # Number of input channels, 0 = first block.
		tmp_channels,                       # Number of intermediate channels.
		out_channels,                       # Number of output channels.
		resolution,                         # Resolution of this block.
		img_channels,                       # Number of input color channels.
		first_layer_idx,                    # Index of the first layer.
		architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
		activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
		resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
		use_fp16            = False,        # Use FP16 for this block?
		fp16_channels_last  = False,        # Use channels-last memory format with FP16?
		freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
		conv_kernel			= 3,
		freq_dist			= 'uniform',
		max_freq			= 512,
		dist_init			= None,
		sort_dist			= True,
	):
		assert in_channels in [0, tmp_channels]
		assert architecture in ['orig', 'skip', 'resnet']
		super().__init__()
		self.in_channels = in_channels
		self.resolution = resolution
		self.img_channels = img_channels
		self.first_layer_idx = first_layer_idx
		self.architecture = architecture
		self.use_fp16 = use_fp16
		self.channels_last = (use_fp16 and fp16_channels_last)
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

		self.num_layers = 0
		def trainable_gen():
			while True:
				layer_idx = self.first_layer_idx + self.num_layers
				trainable = (layer_idx >= freeze_layers)
				self.num_layers += 1
				yield trainable
		trainable_iter = trainable_gen()

		if in_channels == 0 or architecture == 'skip':
			self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
				trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

		self.kernel_size = conv_kernel
		self.conv_clamp = conv_clamp
		self.activation = activation
		self.act_gain = bias_act.activation_funcs[activation].def_gain

		self.conv0_weight = SynthesisGroupKernel(tmp_channels, tmp_channels, sampling_rate=self.resolution
			,freq_dist=freq_dist, max_freq=max_freq, dist_init=dist_init, sort_dist=sort_dist)
		self.conv0_bias = torch.nn.Parameter(torch.zeros([tmp_channels]))
		self.conv1_weight = SynthesisGroupKernel(tmp_channels, out_channels, sampling_rate=self.resolution
			,freq_dist=freq_dist, max_freq=max_freq, dist_init=dist_init, sort_dist=sort_dist)
		self.conv1_bias = torch.nn.Parameter(torch.zeros([out_channels]))
		self.skip_weight = SynthesisGroupKernel(tmp_channels, out_channels, sampling_rate=self.resolution
			,freq_dist=freq_dist, max_freq=max_freq, dist_init=dist_init, sort_dist=sort_dist)

		# if architecture == 'resnet':
		# 	self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
		# 		trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

	def forward(self, x, img, force_fp32=False, update_emas=False):
		if (x if x is not None else img).device.type != 'cuda':
			force_fp32 = True
		dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
		memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

		# Input.
		if x is not None:
			misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
			x = x.to(dtype=dtype, memory_format=memory_format)

		# FromRGB.
		if self.in_channels == 0 or self.architecture == 'skip':
			misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
			img = img.to(dtype=dtype, memory_format=memory_format)
			y = self.fromrgb(img)
			x = x + y if x is not None else y
			img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

		# Main layers.
		if self.architecture == 'resnet':
			act_clamp = self.conv_clamp if self.conv_clamp is not None else None

			skip_w = self.skip_weight(device=x.device, ks=1, update_emas=update_emas).to(x.dtype)
			y = conv2d_resample.conv2d_resample(x=x, w=skip_w, padding=0, f=self.resample_filter, flip_weight=True, down=2)
			y = bias_act.bias_act(x=y, b=None, clamp=act_clamp, act='linear', gain=np.sqrt(0.5))

			conv0_w = self.conv0_weight(device=x.device, ks=3, update_emas=update_emas).to(dtype=x.dtype)
			conv0_x = conv2d_resample.conv2d_resample(x=x, w=conv0_w, padding=1, f=self.resample_filter, flip_weight=True)
			x = bias_act.bias_act(x=conv0_x, b=self.conv0_bias.to(dtype=x.dtype), clamp=act_clamp, act='lrelu', gain=1)

			conv1_w = self.conv1_weight(device=x.device, ks=3, update_emas=update_emas).to(dtype=x.dtype)
			conv1_x = conv2d_resample.conv2d_resample(x=x, w=conv1_w, padding=1, f=self.resample_filter, down=2, flip_weight=True)
			x = bias_act.bias_act(x=conv1_x, b=self.conv1_bias.to(dtype=x.dtype), clamp=act_clamp, act='lrelu', gain=np.sqrt(0.5))

			x = y.add_(x)

			# outgain = (x.square().mean(dim=[2,3], keepdim=True) + 1e-8).rsqrt().clip(min=0.2, max=5)
			# x = x * outgain
		else:
			x = self.conv0(x)
			x = self.conv1(x)

		assert x.dtype == dtype
		return x, img

	def extra_repr(self):
		return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
	def __init__(self, group_size, num_channels=1):
		super().__init__()
		self.group_size = group_size
		self.num_channels = num_channels

	def forward(self, x):
		N, C, H, W = x.shape
		with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
			G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
		F = self.num_channels
		c = C // F

		y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
		y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
		y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
		y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
		y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
		y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
		y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
		x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
		return x

	def extra_repr(self):
		return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchSpatialLayer(torch.nn.Module):
	def __init__(self, group_size, sampling_rate=4):
		super().__init__()
		self.group_size = group_size
		self.sampling_rate = sampling_rate

	def forward(self, x):
		N, C, H, W = x.shape
		with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
			G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
		
		F = self.sampling_rate

		y = x.reshape(G, -1, H, W)    		# [GnCHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
		y = y - y.mean(dim=1, keepdim=True)               # [GnCHW] Subtract mean over group.
		y = y.square().mean(dim=1, keepdim=True)          # [nCHW]  Calc variance over group.
		y = (y + 1e-8).sqrt()               # [nCHW]  Calc stddev over group.
		y = y.mean(dim=[1], keepdim=True)   # [nHW]     Take average over channels and pixels.
		y = y.reshape(-1, 1, H, W)          # [nF11]   Add missing dimensions.
		y = y.repeat(len(x) // G, 1, 1, 1)            # [NFHW]   Replicate over group and pixels.
		x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
		return x

	def extra_repr(self):
		return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
	def __init__(self,
		in_channels,                    # Number of input channels.
		cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
		resolution,                     # Resolution of this block.
		img_channels,                   # Number of input color channels.
		architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
		mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
		mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
		mbstd_spatial_channel=1,
		activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
		conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
		resample_filter		= [1, 3, 3, 1],
		freq_dist			= 'uniform',
		max_freq			= 512,
		dist_init			= None,
		sort_dist			= True,
	):
		assert architecture in ['orig', 'skip', 'resnet']
		super().__init__()
		self.in_channels = in_channels
		self.cmap_dim = cmap_dim
		self.resolution = resolution
		self.img_channels = img_channels
		self.architecture = architecture
		self.conv_clamp = conv_clamp
		self.activation = activation
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

		if architecture == 'skip':
			self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
		self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None

		self.conv_weight = SynthesisGroupKernel(in_channels + mbstd_num_channels, in_channels, sampling_rate=self.resolution
			,freq_dist=freq_dist, max_freq=max_freq, dist_init=dist_init, sort_dist=sort_dist)
		self.conv_bias = torch.nn.Parameter(torch.zeros([in_channels]))

		self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
		self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

	def forward(self, x, img, cmap, force_fp32=False):
		misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
		_ = force_fp32 # unused
		dtype = torch.float32
		memory_format = torch.contiguous_format

		# FromRGB.
		x = x.to(dtype=dtype, memory_format=memory_format)
		if self.architecture == 'skip':
			misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
			img = img.to(dtype=dtype, memory_format=memory_format)
			x = x + self.fromrgb(img)

		# Main layers.
		if self.mbstd is not None:
			x = self.mbstd(x)
		
		conv_w = self.conv_weight(device=x.device, ks=3).to(dtype=x.dtype)
		conv_x = conv2d_resample.conv2d_resample(x=x, w=conv_w, padding=1, f=self.resample_filter, flip_weight=True)
		x = bias_act.bias_act(x=conv_x, b=self.conv_bias.to(dtype=x.dtype), clamp=self.conv_clamp, act=self.activation, gain=1)
		x = self.fc(x.flatten(1))
		x = self.out(x)

		# Conditioning.
		if self.cmap_dim > 0:
			misc.assert_shape(cmap, [None, self.cmap_dim])
			x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

		assert x.dtype == dtype
		return x

	def extra_repr(self):
		return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
	def __init__(self,
		c_dim,                          # Conditioning label (C) dimensionality.
		img_resolution,                 # Input resolution.
		img_channels,                   # Number of input color channels.
		architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
		channel_base        = 32768,    # Overall multiplier for the number of channels.
		channel_max         = 512,      # Maximum number of channels in any layer.
		num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
		conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
		cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
		freq_dist			= 'uniform',
		max_freq			= 512,
		dist_init			= None,
		sort_dist			= True,
		block_kwargs        = {},       # Arguments for DiscriminatorBlock.
		mapping_kwargs      = {},       # Arguments for MappingNetwork.
		epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
	):
		super().__init__()
		self.c_dim = c_dim
		self.img_resolution = img_resolution
		self.img_resolution_log2 = int(np.log2(img_resolution))
		self.img_channels = img_channels
		self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
		channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
		fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

		if cmap_dim is None:
			cmap_dim = channels_dict[4]
		if c_dim == 0:
			cmap_dim = 0

		common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp,
			freq_dist=freq_dist, max_freq=max_freq, dist_init=dist_init, sort_dist=sort_dist)
		cur_layer_idx = 0
		for res in self.block_resolutions:
			in_channels = channels_dict[res] if res < img_resolution else 0
			tmp_channels = channels_dict[res]
			out_channels = channels_dict[res // 2]
			use_fp16 = (res >= fp16_resolution)
			block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
				first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
			setattr(self, f'b{res}', block)
			cur_layer_idx += block.num_layers
		if c_dim > 0:
			self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
		self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

	def forward(self, img, c, update_emas=False, **block_kwargs):
		_ = update_emas # unused
		x = None
		for res in self.block_resolutions:
			block = getattr(self, f'b{res}')
			x, img = block(x, img, **block_kwargs)

		cmap = None
		if self.c_dim > 0:
			cmap = self.mapping(None, c)
		x = self.b4(x, img, cmap)
		return x

	def extra_repr(self):
		return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------