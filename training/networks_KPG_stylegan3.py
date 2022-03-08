# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

from typing import Iterator
from itertools import chain
from torch.nn.parameter import Parameter
from torchvision.utils import save_image
import numpy as np
import scipy.signal
import scipy.optimize
import torch
import dnnlib
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
	x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
	w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
	s,                  # Style tensor: [batch_size, in_channels]
	demodulate  = True, # Apply weight demodulation?
	padding     = 0,    # Padding: int or [padH, padW]
	input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
	with misc.suppress_tracer_warnings(): # this value will be treated as a constant
		batch_size = int(x.shape[0])
	out_channels, in_channels, kh, kw = w.shape
	misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
	misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
	misc.assert_shape(s, [batch_size, in_channels]) # [NI]

	# Pre-normalize inputs.
	if demodulate:
		w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
		s = s * s.square().mean().rsqrt()

	# Modulate weights.
	w = w.unsqueeze(0) # [NOIkk]
	w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

	# Demodulate weights.
	if demodulate:
		dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
		w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

	# Apply input scaling.
	if input_gain is not None:
		input_gain = input_gain.expand(batch_size, in_channels) # [NI]
		w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

	# Execute as one fused op using grouped convolution.
	x = x.reshape(1, -1, *x.shape[2:])
	w = w.reshape(-1, in_channels, kh, kw)
	x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
	x = x.reshape(batch_size, -1, *x.shape[2:])
	return x

#----------------------------------------------------------------------------

@misc.profiled_function
def simple_conv2d(
	x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
	w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
	padding     = 0,    # Padding: int or [padH, padW]
	input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
	with misc.suppress_tracer_warnings(): # this value will be treated as a constant
		batch_size = int(x.shape[0])
	_, out_channels, in_channels, kh, kw = w.shape
	misc.assert_shape(w, [batch_size, out_channels, in_channels, kh, kw]) # [OIkk]
	misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]

	# Apply input scaling.
	if input_gain is not None:
		input_gain = input_gain.expand(batch_size, in_channels) # [NI]
		w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

	# Execute as one fused op using grouped convolution.
	x = x.reshape(1, -1, *x.shape[2:])
	w = w.reshape(-1, in_channels, kh, kw)
	x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
	x = x.reshape(batch_size, -1, *x.shape[2:])
	return x

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

	def forward(self, x, gain=1, down=None):
		curr_down = down if down != None else self.down
		w = self.weight * self.weight_gain
		b = self.bias.to(x.dtype) if self.bias is not None else None
		flip_weight = (self.up == 1) # slightly faster
		x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=curr_down, padding=self.padding, flip_weight=flip_weight)

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

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
	def __init__(self,
		w_dim,          # Intermediate latent (W) dimensionality.
		channels,       # Number of output channels.
		size,           # Output spatial size: int or [width, height].
		sampling_rate,  # Output sampling rate.
		bandwidth,      # Output bandwidth.
	):
		super().__init__()
		self.w_dim = w_dim
		self.channels = channels
		self.size = np.broadcast_to(np.asarray(size), [2])
		self.sampling_rate = sampling_rate
		self.bandwidth = bandwidth

		# Draw random frequencies from uniform 2D disc.
		freqs = torch.randn([self.channels, 2])
		radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
		freqs /= radii * radii.square().exp().pow(0.25)
		freqs *= bandwidth
		phases = torch.rand([self.channels]) - 0.5

		# Setup parameters and buffers.
		self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
		self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
		self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
		self.register_buffer('freqs', freqs)
		self.register_buffer('phases', phases)

	def forward(self, w):
		# Introduce batch dimension.
		transforms = self.transform.unsqueeze(0) # [batch, row, col]
		freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
		phases = self.phases.unsqueeze(0) # [batch, channel]

		# Apply learned transformation.
		t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
		t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
		m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
		m_r[:, 0, 0] = t[:, 0]  # r'_c
		m_r[:, 0, 1] = -t[:, 1] # r'_s
		m_r[:, 1, 0] = t[:, 1]  # r'_s
		m_r[:, 1, 1] = t[:, 0]  # r'_c
		m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
		m_t[:, 0, 2] = -t[:, 2] # t'_x
		m_t[:, 1, 2] = -t[:, 3] # t'_y
		transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

		# Transform frequencies.
		phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
		freqs = freqs @ transforms[:, :2, :2]

		# Dampen out-of-band frequencies that may occur due to the user-specified transform.
		amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

		# Construct sampling grid.
		theta = torch.eye(2, 3, device=w.device)
		theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
		theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
		grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

		# Compute Fourier features.
		x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
		x = x + phases.unsqueeze(1).unsqueeze(2)
		x = torch.sin(x * (np.pi * 2))
		x = x * amplitudes.unsqueeze(1).unsqueeze(2)

		# Apply trainable mapping.
		weight = self.weight / np.sqrt(self.channels)
		x = x @ weight.t()

		# Ensure correct shape.
		x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
		misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
		return x

	def extra_repr(self):
		return '\n'.join([
			f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
			f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])

#----------------------------------------------------------------------------
class SynthesisKernel(torch.nn.Module):
	def __init__(self,
		in_channels,
		out_channels,
		ks,
		sampling_rate,
		bandlimit = None,
		freq_dim = 32,
		butterN = 4,
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.ks = ks
		self.sampling_rate = sampling_rate
		self.bandlimit = max(sampling_rate) // 2 if bandlimit is None else bandlimit
		self.freq_dim = freq_dim * len(sampling_rate)
		
		# Create uniform distribution on disk
		# freqs = torch.randn([self.freq_dim, 2])
		# radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
		# freqs /= radii * radii.square().exp().pow(0.25)
		# freqs *= self.bandlimit

		radii = (torch.rand(self.freq_dim, 1)) * self.bandlimit
		angle = (torch.rand(self.freq_dim, 1) - 0.5) * 2 * np.pi
		freqs = torch.cat([radii * angle.sin(), radii * angle.cos()], dim=1)

		phases = (torch.rand(self.freq_dim) - 0.5)
		_, freq_idx = torch.sort(freqs.norm(dim=-1))
		# self.freqs = torch.nn.Parameter(freqs)
		# self.phases = torch.nn.Parameter(phases)
		self.register_buffer('freqs', freqs[freq_idx])
		self.register_buffer('phases', phases)
		# self.in_weight = torch.nn.Parameter(torch.randn([in_channels, self.freq_dim]))
		# self.out_weight = torch.nn.Parameter(torch.randn([out_channels, self.freq_dim]))
		self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, self.freq_dim]))
		self.butterN = butterN
		self.gain = np.sqrt(1 / (in_channels * (self.ks **2)))

	def forward(self, target_sampling_rate, device, alpha=None):
		if alpha == None:
			alpha = 1
		# Sample signal
		sample_size = self.ks
		freqs = self.freqs.unsqueeze(0)
		phases = self.phases.unsqueeze(0)

		theta = torch.eye(2, 3, device=device)
		theta[0, 0] = 0.5 * 3 / target_sampling_rate
		theta[1, 1] = 0.5 * 3 / target_sampling_rate
		grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, sample_size, sample_size], align_corners=False)
		grids += 0.5 * 3 / target_sampling_rate

		x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
		x = x + phases.unsqueeze(1).unsqueeze(2)
		x = torch.sin(x * (np.pi * 2)).squeeze(0)

		#Compute cutoff frequency for butterworth filter based on alpha
		low_filter = 1
		high_filter = 1
		freq_norm = freqs.norm(dim=-1)

		if target_sampling_rate != min(self.sampling_rate):
			low_cutoff = target_sampling_rate // 4
			low_filter = (1 / (1 + (freq_norm / low_cutoff) ** (-2 * self.butterN))) 

		# high_cutoff = alpha * target_sampling_rate//2 + (1 - alpha) * (target_sampling_rate // 4)
		high_cutoff = target_sampling_rate//2 
		high_filter = (1 / (1 + (freq_norm / high_cutoff) ** (2 * self.butterN))) 
		curr_filter = high_filter * low_filter
		len_freq = self.freq_dim
		freq_idx = (curr_filter > 1e-5).nonzero()[:,1]

		x = x[...,freq_idx] * curr_filter[...,freq_idx]
		w = self.weight[...,freq_idx]
		x = torch.einsum('hwc,oic->oihw', x, w) / np.sqrt(len_freq)
		
		assert torch.isfinite(x).all().item()
		return x * self.gain
	
#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
	def __init__(self,
		# General specifications.
		w_dim,                          # Intermediate latent (W) dimensionality.
		is_torgb,                       # Is this the final ToRGB layer?
		is_critically_sampled,          # Does this layer use critical sampling?
		in_channels,                    # Number of input channels.
		out_channels,                   # Number of output channels.
		target_resolutions,             # Target resolution list.

		# Per resolution specifications.
		use_fp16,                       # Does this layer use FP16?
		in_size,                        # Input spatial size: int or [width, height].
		out_size,                       # Output spatial size: int or [width, height].
		in_sampling_rate,               # Input sampling rate (s).
		out_sampling_rate,              # Output sampling rate (s).
		in_cutoff,                      # Input cutoff frequency (f_c).
		out_cutoff,                     # Output cutoff frequency (f_c).
		in_half_width,                  # Input transition band half-width (f_h).
		out_half_width,                 # Output Transition band half-width (f_h).

		# Hyperparameters.
		conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
		filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
		lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
		use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
		conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
		magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
	):
		super().__init__()
		self.w_dim = w_dim
		self.is_torgb = is_torgb
		self.is_critically_sampled = is_critically_sampled
		self.use_fp16 = use_fp16
		self.in_channels = in_channels
		self.out_channels = out_channels
		# self.in_size = np.broadcast_to(np.asarray(in_size), [2])
		# self.out_size = np.broadcast_to(np.asarray(out_size), [2])
		self.in_size = in_size
		self.out_size = out_size
		self.in_sampling_rate = in_sampling_rate
		self.out_sampling_rate = out_sampling_rate
		# self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
		self.in_cutoff = in_cutoff
		self.out_cutoff = out_cutoff
		self.in_half_width = in_half_width
		self.out_half_width = out_half_width
		self.conv_kernel = 1 if is_torgb else conv_kernel
		self.conv_clamp = conv_clamp
		self.magnitude_ema_beta = magnitude_ema_beta

		# Alias filter initialization
		self.filter_size = filter_size
		self.lrelu_upsampling = lrelu_upsampling
		self.use_radial_filters = use_radial_filters
		
		self.padding = {}
		self.up_factor = {}
		self.down_factor = {}
		for res in target_resolutions:
			up_filter, down_filter, filter_args = self.get_filter(
				in_sampling_rate=in_sampling_rate[res], out_sampling_rate=out_sampling_rate[res],
				in_cutoff=in_cutoff[res], out_cutoff=out_cutoff[res],
				in_half_width=in_half_width[res], out_half_width=out_half_width[res], tmp_rate=(1 if self.is_torgb else self.lrelu_upsampling))
			padding = self.get_down_padding(in_size=in_size[res], out_size=out_size[res], **filter_args)
			self.register_buffer(f'up_filter_{res}', up_filter)
			self.register_buffer(f'down_filter_{res}', down_filter)
			self.padding[res] = padding
			self.up_factor[res] = filter_args['up_factor']
			self.down_factor[res] = filter_args['down_factor']

		self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
		self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, self.conv_kernel, self.conv_kernel])) 
		self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
		self.register_buffer('magnitude_ema', torch.ones([]))

		
	def get_filter(self, 
		in_sampling_rate, 
		out_sampling_rate, 
		in_cutoff,
		out_cutoff,
		in_half_width,
		out_half_width,
		tmp_rate,
		filter_size = None,
	):
		tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * tmp_rate
		filter_size = filter_size if filter_size is not None else self.filter_size

		# Design upsampling filter.
		up_factor = int(np.rint(tmp_sampling_rate / in_sampling_rate))
		assert in_sampling_rate * up_factor == tmp_sampling_rate
		up_taps = filter_size * up_factor if up_factor > 1 and not self.is_torgb else 1
		up_filter = self.design_lowpass_filter(numtaps=up_taps, cutoff=in_cutoff, width=in_half_width*2, fs=tmp_sampling_rate)

		# Design downsampling filter.
		down_factor = int(np.rint(tmp_sampling_rate / out_sampling_rate))
		assert out_sampling_rate * down_factor == tmp_sampling_rate
		down_taps = filter_size * down_factor if down_factor > 1 and not self.is_torgb else 1
		down_radial = self.use_radial_filters and not self.is_critically_sampled
		down_filter = self.design_lowpass_filter(numtaps=down_taps, cutoff=out_cutoff, width=out_half_width*2, fs=tmp_sampling_rate, radial=down_radial)

		filter_args = {
			"up_factor": up_factor,
			"down_factor": down_factor,
			"up_taps": up_taps,
			"down_taps": down_taps,
		}

		return up_filter, down_filter, filter_args
	
	def get_down_padding(self,
		in_size,
		out_size,
		up_factor,
		down_factor,
		up_taps,
		down_taps,
	):
		# Compute padding.
		pad_total = (out_size - 1) * down_factor + 1 # Desired output size before downsampling.
		pad_total -= (in_size + self.conv_kernel - 1) * up_factor # Input size after upsampling.
		pad_total += up_taps + down_taps - 2 # Size reduction caused by the filters.
		pad_lo = (pad_total + up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
		pad_hi = pad_total - pad_lo
		padding = [int(pad_lo), int(pad_hi), int(pad_lo), int(pad_hi)]
		return padding

	def forward(self, x, w, resolution=None, alpha=None, noise_mode='random', force_fp32=False, update_emas=False):
		assert noise_mode in ['random', 'const', 'none'] # unused
		assert (resolution is not None) and (alpha is not None)
		misc.assert_shape(x, [None, self.in_channels, int(self.in_size[resolution]), int(self.in_size[resolution])])
		misc.assert_shape(w, [x.shape[0], self.w_dim])

		# Track input magnitude.
		if update_emas:
			with torch.autograd.profiler.record_function('update_magnitude_ema'):
				magnitude_cur = x.detach().to(torch.float32).square().mean()
				self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
		input_gain = self.magnitude_ema.rsqrt()

		# Execute affine layer.
		styles = self.affine(w)
		if self.is_torgb:
			weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
			styles = styles * weight_gain

		# Execute modulated conv2d.
		dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
		x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
			padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)
		
		# Execute bias, filtered leaky ReLU, and clamping.
		gain = 1 if self.is_torgb else np.sqrt(2)
		slope = 1 if self.is_torgb else 0.2
		x = filtered_lrelu.filtered_lrelu(x=x, fu=getattr(self, f'up_filter_{resolution}'), fd=getattr(self, f'down_filter_{resolution}'), 
			b=self.bias.to(x.dtype), up=self.up_factor[resolution], down=self.down_factor[resolution], padding=self.padding[resolution], 
			gain=gain, slope=slope, clamp=self.conv_clamp)

		# Ensure correct shape and dtype.
		misc.assert_shape(x, [None, self.out_channels, int(self.out_size[resolution]), int(self.out_size[resolution])])
		assert x.dtype == dtype
		return x

	@staticmethod
	def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
		assert numtaps >= 1

		# Identity filter.
		if numtaps == 1:
			return None

		# Separable Kaiser low-pass filter.
		if not radial:
			f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
			return torch.as_tensor(f, dtype=torch.float32)

		# Radially symmetric jinc-based filter.
		x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
		r = np.hypot(*np.meshgrid(x, x))
		f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
		beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
		w = np.kaiser(numtaps, beta)
		f *= np.outer(w, w)
		f /= np.sum(f)
		return torch.as_tensor(f, dtype=torch.float32)

	def extra_repr(self):
		return '\n'.join([
			f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
			f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
			f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
			f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
			f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
			f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
			f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
	def __init__(self,
		# General specifications.
		w_dim,                          # Intermediate latent (W) dimensionality.
		is_torgb,                       # Is this the final ToRGB layer?
		is_critically_sampled,          # Does this layer use critical sampling?
		in_channels,                    # Number of input channels.
		out_channels,                   # Number of output channels.
		target_resolutions,             # Target resolution list.

		# Per resolution specifications.
		use_fp16,                       # Does this layer use FP16?
		in_size,                        # Input spatial size: int or [width, height].
		out_size,                       # Output spatial size: int or [width, height].
		in_sampling_rate,               # Input sampling rate (s).
		out_sampling_rate,              # Output sampling rate (s).
		in_cutoff,                      # Input cutoff frequency (f_c).
		out_cutoff,                     # Output cutoff frequency (f_c).
		in_half_width,                  # Input transition band half-width (f_h).
		out_half_width,                 # Output Transition band half-width (f_h).

		# Hyperparameters.
		conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
		filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
		lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
		use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
		conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
		magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
		layer_idx			= 0,		# For Debug
	):
		super().__init__()
		self.w_dim = w_dim
		self.is_torgb = is_torgb
		self.is_critically_sampled = is_critically_sampled
		self.use_fp16 = use_fp16
		self.in_channels = in_channels
		self.out_channels = out_channels
		# self.in_size = np.broadcast_to(np.asarray(in_size), [2])
		# self.out_size = np.broadcast_to(np.asarray(out_size), [2])
		self.in_size = in_size
		self.out_size = out_size
		self.in_sampling_rate = in_sampling_rate
		self.out_sampling_rate = out_sampling_rate
		# self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
		self.in_cutoff = in_cutoff
		self.out_cutoff = out_cutoff
		self.in_half_width = in_half_width
		self.out_half_width = out_half_width
		self.conv_kernel = 1 if is_torgb else conv_kernel
		self.conv_clamp = conv_clamp
		self.magnitude_ema_beta = magnitude_ema_beta

		self.layer_idx = layer_idx

		# Alias filter initialization
		self.filter_size = filter_size
		self.lrelu_upsampling = lrelu_upsampling
		self.use_radial_filters = use_radial_filters
		
		self.padding = {}
		self.up_factor = {}
		self.down_factor = {}
		for res in target_resolutions:
			up_filter, down_filter, filter_args = self.get_filter(
				in_sampling_rate=in_sampling_rate[res], out_sampling_rate=out_sampling_rate[res],
				in_cutoff=in_cutoff[res], out_cutoff=out_cutoff[res],
				in_half_width=in_half_width[res], out_half_width=out_half_width[res], tmp_rate=(1 if self.is_torgb else self.lrelu_upsampling))
			padding = self.get_down_padding(in_size=in_size[res], out_size=out_size[res], **filter_args)
			self.register_buffer(f'up_filter_{res}', up_filter)
			self.register_buffer(f'down_filter_{res}', down_filter)
			self.padding[res] = padding
			self.up_factor[res] = filter_args['up_factor']
			self.down_factor[res] = filter_args['down_factor']

		# Path initialization
		self.init_path()
		self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
		self.register_buffer('magnitude_ema', torch.ones([]))
		self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
		# self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, 1, 1]))
		self.weight_gen = SynthesisKernel(in_channels=self.in_channels, out_channels=out_channels, 
										  ks=self.conv_kernel, sampling_rate=self.target_sr)
		
	def get_filter(self, 
		in_sampling_rate, 
		out_sampling_rate, 
		in_cutoff,
		out_cutoff,
		in_half_width,
		out_half_width,
		tmp_rate,
		filter_size = None,
	):
		tmp_sampling_rate= max(in_sampling_rate, out_sampling_rate) * tmp_rate
		filter_size = filter_size if filter_size is not None else self.filter_size

		# Design upsampling filter.
		up_factor = int(np.rint(tmp_sampling_rate / in_sampling_rate))
		assert in_sampling_rate * up_factor == tmp_sampling_rate 
		up_taps = filter_size * up_factor if up_factor > 1 and not self.is_torgb else 1
		up_filter = self.design_lowpass_filter(numtaps=up_taps, cutoff=in_cutoff, width=in_half_width*2, fs=tmp_sampling_rate)

		# Design downsampling filter.
		down_factor = int(np.rint(tmp_sampling_rate/ out_sampling_rate))
		assert out_sampling_rate * down_factor == tmp_sampling_rate 
		down_taps = filter_size * down_factor if down_factor > 1 and not self.is_torgb else 1
		down_radial = self.use_radial_filters and not self.is_critically_sampled
		down_filter = self.design_lowpass_filter(numtaps=down_taps, cutoff=out_cutoff, width=out_half_width*2, fs=tmp_sampling_rate, radial=down_radial)

		filter_args = {
			"up_factor": up_factor,
			"down_factor": down_factor,
			"up_taps": up_taps,
			"down_taps": down_taps,
		}

		return up_filter, down_filter, filter_args
	
	def get_down_padding(self,
		in_size,
		out_size,
		up_factor,
		down_factor,
		up_taps,
		down_taps,
	):
		# Compute padding.
		pad_total = (out_size - 1) * down_factor + 1 # Desired output size before downsampling.
		pad_total -= (in_size + self.conv_kernel - 1) * up_factor # Input size after upsampling.
		pad_total += up_taps + down_taps - 2 # Size reduction caused by the filters.
		pad_lo = (pad_total + up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
		pad_hi = pad_total - pad_lo
		padding = [int(pad_lo), int(pad_hi), int(pad_lo), int(pad_hi)]
		return padding

	def get_up_padding(self,
		in_size,
		out_size,
		up_factor,
		down_factor,
		up_taps,
		down_taps,
	):
		# Compute padding.
		tmp_size = (in_size // down_factor + self.conv_kernel - 1)
		pad_total = (out_size - tmp_size * up_factor)
		pad_lo = (pad_total) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
		pad_hi = pad_total - pad_lo
		padding = [int(pad_lo), int(pad_hi), int(pad_lo), int(pad_hi)]
		return padding
	
	def init_path(self):
		paths = {}
		target_resolution = list(self.in_sampling_rate.keys())
		target_sr = sorted(list(set(self.in_sampling_rate.values())))
		self.target_sr = target_sr
		target_sr_res = []
		min_sr = min(target_sr)

		# Find maximal resolution 
		for sampling_rate in target_sr:
			max_res = 0
			for res, res_sr in self.in_sampling_rate.items():
				if (sampling_rate == res_sr) and (res > max_res):
					max_res = res
			target_sr_res.append(max_res)

		for res in target_resolution:
			cur_res_path = []
			for sampling_rate, max_res in zip(target_sr, target_sr_res):
				if self.in_sampling_rate[res] < sampling_rate:
					break

				up_filter_name = None,
				down_filter_name = None,
				up_args = None
				down_args = None
				weight_args = None
				
				down_factor = self.in_sampling_rate[res] // sampling_rate
				if down_factor > 1:
					# up_filter, down_filter, filter_args = self.get_filter(
					# 	in_sampling_rate=self.out_sampling_rate[max_res], out_sampling_rate=self.in_sampling_rate[max_res],
					# 	in_cutoff=self.out_cutoff[max_res], out_cutoff=self.in_cutoff[max_res],
					# 	in_half_width=self.out_half_width[max_res], out_half_width=self.in_half_width[max_res],
					# 	filter_size=3 * down_factor, tmp_rate=2
					# )
					up_filter = self.design_lowpass_filter(numtaps=down_factor * 6, cutoff=sampling_rate//2, width=(np.sqrt(2) - 1)*sampling_rate, 
									fs = self.in_size[res] + self.conv_kernel - 1)
					down_filter = self.design_lowpass_filter(numtaps=down_factor * 6, cutoff=sampling_rate//2, width=(np.sqrt(2) - 1)*sampling_rate, 
									fs = self.in_size[res])
					padding = self.get_up_padding(in_size=self.in_size[res], out_size=self.in_size[res] + self.conv_kernel - 1,
								up_factor=down_factor, down_factor=down_factor, up_taps=None, down_taps=None)

					self.register_buffer(f'up_filter_{res}_{sampling_rate}_{down_factor}', up_filter)
					self.register_buffer(f'down_filter_{res}_{sampling_rate}_{down_factor}', down_filter)
					
					up_filter_name = f'up_filter_{res}_{sampling_rate}_{down_factor}'
					down_filter_name = f'down_filter_{res}_{sampling_rate}_{down_factor}'
					# up_filter_name = f'up_filter_{max_res}'
					# down_filter_name = f'down_filter_{max_res}'
					up_args = {
						"up": down_factor,
						"padding": padding,
					}
					down_args = {
						"down": down_factor,
					}

				weight_args = {
					"target_sampling_rate" : sampling_rate,
				}
				cur_sample_path = dnnlib.EasyDict(
					path=f'{res}_{sampling_rate}',
					affine_name=f'affine_{sampling_rate}',
					up_filter=up_filter_name,
					down_filter=down_filter_name,
					up_args=up_args,
					down_args=down_args,
					weight_args=weight_args,
					use_fp16=self.use_fp16[sampling_rate]
				)
				cur_res_path.append(cur_sample_path)

			paths[res] = cur_res_path
		self.paths = paths
				

	def forward(self, x, w, resolution=None, alpha=None, noise_mode='random', force_fp32=False, update_emas=False):
		assert noise_mode in ['random', 'const', 'none'] # unused
		assert (resolution is not None) and (alpha is not None)
		misc.assert_shape(x, [None, self.in_channels, int(self.in_size[resolution]), int(self.in_size[resolution])])
		misc.assert_shape(w, [x.shape[0], self.w_dim])

		# Track input magnitude.
		if update_emas:
			with torch.autograd.profiler.record_function('update_magnitude_ema'):
				magnitude_cur = x.detach().to(torch.float32).square().mean()
				self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
		input_gain = self.magnitude_ema.rsqrt()

		path_x = {}
		curr_style = self.affine(w)
		# skip_x = modulated_conv2d(x=x, w=self.weight, s=curr_style, padding=1, demodulate=(not self.is_torgb), input_gain=input_gain)
		alpha_gain = []
		for path_id, path_args in enumerate(self.paths[resolution]):
			path_x[path_args.path] = x

			if path_args.down_args:
				down_filter = getattr(self, path_args.down_filter)
				path_x[path_args.path] = upfirdn2d.downsample2d(path_x[path_args.path], f=down_filter, **path_args.down_args)

			# Execute affine layer.

			if self.is_torgb:
				weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
				styles = styles * weight_gain

			dtype = torch.float16 if (path_args.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

			curr_alpha = alpha.item() if (len(self.paths[resolution]) > 1) and (path_id == len(self.paths[resolution]) - 1) else 1
			# Generate weight given alpha
			weight = self.weight_gen(alpha=curr_alpha, device=x.device, **path_args.weight_args)

			# Execute modulated conv2d.
			path_x[path_args.path] = modulated_conv2d(x=path_x[path_args.path].to(dtype), w=weight, s=curr_style,
				padding=self.conv_kernel-1, input_gain=input_gain * curr_alpha) 
			alpha_gain.append(curr_alpha ** 2)
			
			if path_args.up_args:
				up_filter = getattr(self, path_args.up_filter)
				path_x[path_args.path] = upfirdn2d.upsample2d(path_x[path_args.path], f=up_filter, impl="ref", **path_args.up_args)
				# t_size = x.shape[-1] + self.conv_kernel - 1
				# path_x[path_args.path] = torch.nn.functional.interpolate(path_x[path_args.path], mode="bilinear", size=[t_size, t_size])

			# save_image(path_x[path_args.path][0,:3],f'{resolution}_{self.layer_idx}_{path_args.path}.png') 
		
		x = (sum(path_x.values()).to(dtype)) / np.sqrt(4 * sum(alpha_gain))


		# Execute bias, filtered leaky ReLU, and clamping.
		gain = 1 if self.is_torgb else np.sqrt(2)
		slope = 1 if self.is_torgb else 0.2
		x = filtered_lrelu.filtered_lrelu(x=x, fu=getattr(self, f'up_filter_{resolution}'), fd=getattr(self, f'down_filter_{resolution}'), 
			b=self.bias.to(x.dtype), up=self.up_factor[resolution], down=self.down_factor[resolution], padding=self.padding[resolution], 
			gain=gain, slope=slope, clamp=self.conv_clamp)

		# Ensure correct shape and dtype.
		misc.assert_shape(x, [None, self.out_channels, int(self.out_size[resolution]), int(self.out_size[resolution])])
		assert x.dtype == dtype
		return x

	@staticmethod
	def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
		assert numtaps >= 1

		# Identity filter.
		if numtaps == 1:
			return None

		# Separable Kaiser low-pass filter.
		if not radial:
			f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
			return torch.as_tensor(f, dtype=torch.float32)

		# Radially symmetric jinc-based filter.
		x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
		r = np.hypot(*np.meshgrid(x, x))
		f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
		beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
		w = np.kaiser(numtaps, beta)
		f *= np.outer(w, w)
		f /= np.sum(f)
		return torch.as_tensor(f, dtype=torch.float32)

	def extra_repr(self):
		return '\n'.join([
			f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
			f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
			f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
			f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
			f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
			f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
			f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
	def __init__(self,
		w_dim,                          # Intermediate latent (W) dimensionality.
		target_resolutions,             # Target resolution lists
		img_channels,                   # Number of color channels.
		channel_base        = 32768,    # Overall multiplier for the number of channels.
		channel_max         = 512,      # Maximum number of channels in any layer.
		channel_scale       = 1, 
		num_layers          = 12,       # Total number of layers, excluding Fourier features and ToRGB.
		num_critical        = 0,        # Number of critically sampled layers at the end.
		first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
		first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
		last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
		margin_size         = 10,       # Number of additional pixels outside the image.
		output_scale        = 0.25,     # Scale factor for the output image.
		num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
		alpha_schedule      = 1e-5,     # per iteration alpha scheduler
		**layer_kwargs,                 # Arguments for SynthesisLayer.
	):
		super().__init__()
		self.w_dim = w_dim
		self.num_ws = num_layers + 2
		self.target_resolutions = sorted(target_resolutions)
		self.curr_resolution = target_resolutions[0]
		self.prev_resolution = None
		self.img_channels = img_channels
		self.num_layers = num_layers
		self.num_critical = num_critical
		self.margin_size = margin_size
		self.output_scale = output_scale
		self.num_fp16_res = num_fp16_res
		self.register_buffer("alpha", torch.ones([]))
		self.alpha_schedule = alpha_schedule
		self.last_stopband_rel = last_stopband_rel
		self.first_cutoff = first_cutoff
		self.first_stopband = first_stopband

		band_args_dict = {k: self.compute_band(k) for k in self.target_resolutions}
		channels = np.rint(np.minimum((channel_base * channel_scale / 2) / band_args_dict[max(self.target_resolutions)].cutoffs, 
										channel_max * channel_scale))

		input_band_args, self.per_layer_band_args = self.permute_per_layer_band(band_args_dict)
		
		# Construct layers.
		self.input = SynthesisInput(
			w_dim=self.w_dim, channels=int(channels[0]), **input_band_args)
		self.layer_names = []
		for idx in range(self.num_layers):
			prev = max(idx - 1, 0)
			is_critically_sampled = (idx >= self.num_layers - self.num_critical)
			layer = SynthesisLayer(
				layer_idx=idx, w_dim=self.w_dim, is_torgb=False, is_critically_sampled=is_critically_sampled,
				in_channels=int(channels[prev]), out_channels= int(channels[idx]), target_resolutions=target_resolutions,
				**self.per_layer_band_args[idx],
				**layer_kwargs
			)
			name = f'L{idx}_{layer.out_channels}'
			setattr(self, name, layer)
			self.layer_names.append(name)
		
		# Add toRGB layer
		layer = ToRGBLayer(w_dim=self.w_dim, is_torgb=True, is_critically_sampled=True, in_channels=int(channels[-1]), out_channels=self.img_channels,
				target_resolutions=target_resolutions, **self.per_layer_band_args[-1], **layer_kwargs)
		name = f'L{idx+1}_{layer.out_channels}'
		setattr(self, name, layer)
		self.layer_names.append(name)
	
	def permute_per_layer_band(self, band_args_dict):
		per_layer_band_args = []
		# Add input band args
		max_res = max(self.target_resolutions)
		input_band_args = dnnlib.EasyDict(
			size=int(band_args_dict[max_res].sizes[0]),
			sampling_rate=int(band_args_dict[max_res].sampling_rates[0]),
			bandwidth=band_args_dict[max_res].cutoffs[0],
		)

		# Add per layer band args
		for idx in range(self.num_layers + 1):
			prev = max(idx - 1, 0)
			idx_band_args = dnnlib.EasyDict(
				use_fp16={}, in_size={}, out_size={},
				in_sampling_rate={}, out_sampling_rate={}, 
				in_cutoff={}, out_cutoff={},
				in_half_width={}, out_half_width={}
			)
			for res, args in band_args_dict.items():
				use_fp16 = (args.sampling_rates[idx] * (2 ** self.num_fp16_res) > self.target_resolutions[-1])
				idx_band_args.use_fp16[res] = use_fp16
				idx_band_args.in_size[res] = int(args.sizes[prev])
				idx_band_args.out_size[res] = int(args.sizes[idx])
				idx_band_args.in_sampling_rate[res] = int(args.sampling_rates[prev])
				idx_band_args.out_sampling_rate[res] = int(args.sampling_rates[idx])
				idx_band_args.in_cutoff[res] = args.cutoffs[prev] 
				idx_band_args.out_cutoff[res] = args.cutoffs[idx]
				idx_band_args.in_half_width[res] = args.half_widths[prev] 
				idx_band_args.out_half_width[res] = args.half_widths[idx]
			
			# TODO: Manual setting, Need to change in future
			idx_band_args.use_fp16[16] = False

			per_layer_band_args.append(idx_band_args)
		
		return input_band_args, per_layer_band_args

	def compute_band(self, img_resolution):
		# Geometric progression of layer cutoffs and min. stopbands.
		last_cutoff = img_resolution / 2 # f_{c,N}
		last_stopband = last_cutoff * self.last_stopband_rel # f_{t,N}
		exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
		cutoffs = self.first_cutoff * (last_cutoff / self.first_cutoff) ** exponents # f_c[i]
		stopbands = self.first_stopband * (last_stopband / self.first_stopband) ** exponents # f_t[i]

		# Compute remaining layer parameters.
		sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, img_resolution)))) # s[i]
		half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
		sizes = sampling_rates + self.margin_size * 2
		sizes[-2:] = img_resolution

		return dnnlib.EasyDict(
			cutoffs=cutoffs,
			stopbands=stopbands,
			half_widths=half_widths,
			sampling_rates=sampling_rates,
			sizes=sizes
		)

	def forward(self, ws, **layer_kwargs):
		misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
		ws = ws.to(torch.float32).unbind(dim=1)

		# Execute layers.
		x = self.input(ws[0])
		for name, w in zip(self.layer_names, ws[1:]):
			x = getattr(self, name)(x, w, resolution=self.curr_resolution, alpha=self.alpha, **layer_kwargs)

		if self.output_scale != 1:
			x = x * self.output_scale

		# Ensure correct shape and dtype.
		misc.assert_shape(x, [None, self.img_channels, self.curr_resolution, self.curr_resolution])
		x = x.to(torch.float32)
		return x

	def extra_repr(self):
		return '\n'.join([
			f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
			f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
			f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
			f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])

	def resolution_parameters(self) -> Iterator[Parameter]:
		return self.parameters()
		# Input parameters
		input_params = self.input.parameters()

		# Feature backbone parameters
		cur_res_params = [input_params]
		for layer in self.layer_names:
			cur_res_params.append(getattr(self, layer).parameters())
			if getattr(self, layer).target_resolution == self.target_resolution:
				break
		
		# ToRGB parameters
		if self.prev_resolution:
			cur_res_params.append(getattr(self, self.to_rgb_names[self.prev_resolution]).parameters())
		cur_res_params.append(getattr(self, self.to_rgb_names[self.target_resolution]).parameters())

		return chain(*cur_res_params)
	
#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
	def __init__(self,
		z_dim,                      # Input latent (Z) dimensionality.
		c_dim,                      # Conditioning label (C) dimensionality.
		w_dim,                      # Intermediate latent (W) dimensionality.
		target_resolutions,         # Target resolution lists.
		img_channels,               # Number of output color channels.
		mapping_kwargs      = {},   # Arguments for MappingNetwork.
		**synthesis_kwargs,         # Arguments for SynthesisNetwork.
	):
		super().__init__()
		self.z_dim = z_dim
		self.c_dim = c_dim
		self.w_dim = w_dim
		self.target_resolutions = target_resolutions
		self.img_channels = img_channels
		self.synthesis = SynthesisNetwork(w_dim=w_dim, target_resolutions=target_resolutions, img_channels=img_channels, **synthesis_kwargs)
		self.num_ws = self.synthesis.num_ws
		self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

	def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
		ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
		img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
		return img
	
	def set_resolution(self, cur_resolution, alpha=0):
		self.synthesis.curr_resolution = cur_resolution
		self.synthesis.alpha.copy_(torch.ones([], device=self.synthesis.alpha.device) * alpha)
	
	def resolution_parameters(self):
		return self.parameters()
		ws_params = self.mapping.parameters()
		synth_params = self.synthesis.resolution_parameters()
		return chain(*[ws_params, synth_params])
	
#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
	def __init__(self,
		in_channels,                        # Number of input channels, 0 = first block.
		tmp_channels,                       # Number of intermediate channels.
		out_channels,                       # Number of output channels.
		resolution,                         # Resolution of this block.
		target_resolutions,					# Resolution of Discriminator targets
		img_channels,                       # Number of input color channels.
		first_layer_idx,                    # Index of the first layer.
		architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
		activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
		resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
		use_fp16            = False,        # Use FP16 for this block?
		fp16_channels_last  = False,        # Use channels-last memory format with FP16?
		freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
		frgb                = False,        # For layers which need fromRGB during progressive training
	):
		assert in_channels in [0, tmp_channels]
		assert architecture in ['orig', 'skip', 'resnet']
		super().__init__()
		self.in_channels = in_channels
		self.resolution = resolution
		self.sampling_rates = {s: s if s <= self.resolution else self.resolution for s in target_resolutions}
		self.sampling_rates_list = sorted(list(set(self.sampling_rates.values())))
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

		if in_channels == 0 or architecture == 'skip' or frgb:
			self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
				trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

		self.kernel_size = 3
		self.conv_clamp = conv_clamp
		self.activation = activation
		self.act_gain = bias_act.activation_funcs[activation].def_gain
		self.conv0_weight = SynthesisKernel(tmp_channels, tmp_channels, ks=self.kernel_size, sampling_rate=self.sampling_rates_list)
		self.conv0_bias = torch.nn.Parameter(torch.zeros([tmp_channels]))
		self.conv1_weight = SynthesisKernel(tmp_channels, out_channels, ks=self.kernel_size, sampling_rate=self.sampling_rates_list)
		self.conv1_bias = torch.nn.Parameter(torch.zeros([out_channels]))


		if architecture == 'resnet':
			self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
				trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

	def forward(self, x, img, resolution, alpha=None, force_fp32=False):
		if (x if x is not None else img).device.type != 'cuda':
			force_fp32 = True
		dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
		memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
		sample_size = min(self.resolution, resolution)
		curr_down = 2 if sample_size == self.resolution else 1
		max_sample_rate = self.sampling_rates[resolution]

		# Input.
		if x is not None:
			misc.assert_shape(x, [None, self.in_channels, sample_size, sample_size])
			x = x.to(dtype=dtype, memory_format=memory_format)

		# FromRGB.
		if self.in_channels == 0 or self.architecture == 'skip':
			misc.assert_shape(img, [None, self.img_channels, sample_size, sample_size])
			img = img.to(dtype=dtype, memory_format=memory_format)
			y = self.fromrgb(img)
			x = x + y if x is not None else y
			img = upfirdn2d.downsample2d(img, self.resample_filter) if (self.architecture == 'skip') else None

		# Main layers.
		if self.architecture == 'resnet':
			y = self.skip(x, gain=np.sqrt(0.5), down=curr_down)

			conv0_x = {}
			conv0_w = {}
			alpha_gain = []
			for res in self.sampling_rates_list:
				if res > max_sample_rate:
					break
				curr_alpha = alpha.item() if res == resolution else 1
				alpha_gain.append(curr_alpha ** 2)
				conv0_w[res] = self.conv0_weight(res, x.device, curr_alpha).to(dtype=x.dtype)
				if max_sample_rate // res > 1:
					conv0_x[res] = upfirdn2d.downsample2d(x, self.resample_filter, down=max_sample_rate // res)
					conv0_x[res] = conv2d_resample.conv2d_resample(x=conv0_x[res], w=conv0_w[res], padding=self.kernel_size//2) * curr_alpha
					conv0_x[res] = upfirdn2d.upsample2d(conv0_x[res], self.resample_filter, up=max_sample_rate // res)
				else:
					conv0_x[res] = conv2d_resample.conv2d_resample(x=x, w=conv0_w[res], padding=self.kernel_size//2) * curr_alpha
				conv0_x[res] = conv0_x[res].unsqueeze(0)
			alpha_gain = sum(alpha_gain)
			conv0_x = torch.cat(list(conv0_x.values()), dim=0).sum(dim=0).to(dtype=x.dtype) / np.sqrt(alpha_gain)
			act_clamp = self.conv_clamp if self.conv_clamp is not None else None
			x = bias_act.bias_act(conv0_x, self.conv0_bias.to(dtype=x.dtype), act=self.activation, gain=self.act_gain, clamp=act_clamp)

			conv1_x = {}
			conv1_w = {}
			alpha_gain = []
			for res in self.sampling_rates_list:
				if res > max_sample_rate:
					break
				curr_alpha = alpha.item() if res == resolution else 1
				alpha_gain.append(curr_alpha ** 2)
				conv1_w[res] = self.conv1_weight(res, x.device, curr_alpha).to(dtype=x.dtype)
				if max_sample_rate // res > 1:
					conv1_x[res] = upfirdn2d.downsample2d(x, self.resample_filter, down=max_sample_rate // res)
					conv1_x[res] = conv2d_resample.conv2d_resample(x=conv1_x[res], w=conv1_w[res], padding=self.kernel_size//2) * curr_alpha
					conv1_x[res] = upfirdn2d.upsample2d(conv1_x[res], self.resample_filter, up=max_sample_rate // res)
				else:
					conv1_x[res] = conv2d_resample.conv2d_resample(x=x, w=conv1_w[res], padding=self.kernel_size//2) * curr_alpha
				conv1_x[res] = conv1_x[res].unsqueeze(0)
			alpha_gain = sum(alpha_gain)
			conv1_x = torch.cat(list(conv1_x.values()), dim=0).sum(dim=0).to(dtype=x.dtype) / np.sqrt(alpha_gain)
			conv1_x = upfirdn2d.downsample2d(conv1_x, self.resample_filter, down=curr_down)
			act_clamp = self.conv_clamp if self.conv_clamp is not None else None
			x = bias_act.bias_act(conv1_x, self.conv1_bias.to(dtype=x.dtype), act=self.activation, gain=self.act_gain * 0.5, clamp=act_clamp)
			x = y.add_(x)
		else:
			x = self.conv0(x)
			x = self.conv1(x)

		assert x.dtype == dtype
		return x, img

	def extra_repr(self):
		return f'resolution={self.resolution:d}, architecture={self.architecture:s}'
	
	def conv_parameters(self):
		params = [
			self.conv0.parameters(),
			self.conv1.parameters()
		]
		if self.architecture == "resnet":
			params.append(self.skip.parameters())
		return chain(*params)
	
	def frgb_parameters(self):
		params = []
		if hasattr(self, 'fromrgb'):
			params.append(self.fromrgb.parameters())
		return chain(*params)

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
class DiscriminatorEpilogue(torch.nn.Module):
	def __init__(self,
		in_channels,                    # Number of input channels.
		cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
		resolution,                     # Resolution of this block.
		img_channels,                   # Number of input color channels.
		architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
		mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
		mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
		activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
		conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
	):
		assert architecture in ['orig', 'skip', 'resnet']
		super().__init__()
		self.in_channels = in_channels
		self.cmap_dim = cmap_dim
		self.resolution = resolution
		self.img_channels = img_channels
		self.architecture = architecture

		if architecture == 'skip':
			self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
		self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
		self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
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
		x = self.conv(x)
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
		target_resolutions,             # Target resolution list.
		img_channels,                   # Number of input color channels.
		architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
		channel_base        = 32768,    # Overall multiplier for the number of channels.
		channel_max         = 512,      # Maximum number of channels in any layer.
		num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
		conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
		cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
		alpha_schedule      = 1e-5,     # Schedule rate for alpha
		block_kwargs        = {},       # Arguments for DiscriminatorBlock.
		mapping_kwargs      = {},       # Arguments for MappingNetwork.
		epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
	):
		super().__init__()
		self.c_dim = c_dim
		self.target_resolutions = target_resolutions
		self.max_resolution = max(target_resolutions)
		self.curr_resolution = min(target_resolutions)
		self.prev_resolution = None
		self.max_resolution_log2 = int(np.log2(self.max_resolution))
		self.img_channels = img_channels
		self.block_resolutions = [2 ** i for i in range(self.max_resolution_log2, 2, -1)]
		self.register_buffer("alpha", torch.ones([]))
		self.alpha_schedule = alpha_schedule

		channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
		fp16_resolution = max(2 ** (self.max_resolution_log2 + 1 - num_fp16_res), 8)

		if cmap_dim is None:
			cmap_dim = channels_dict[4]
		if c_dim == 0:
			cmap_dim = 0

		common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
		cur_layer_idx = 0
		min_res = min(self.target_resolutions)

		for res in self.block_resolutions:
			in_channels = channels_dict[res] if res < self.max_resolution else 0
			tmp_channels = channels_dict[res]
			out_channels = channels_dict[res // 2]
			use_fp16 = (res >= fp16_resolution)
			frgb = res >= self.curr_resolution
			block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res, target_resolutions=self.target_resolutions,
				first_layer_idx=cur_layer_idx, use_fp16=use_fp16, frgb=frgb, **block_kwargs, **common_kwargs)
			setattr(self, f'b{res}', block)
			cur_layer_idx += block.num_layers
		if c_dim > 0:
			self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
		self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
 
	def forward(self, img, c, update_emas=False, **block_kwargs):
		x = None
		for res in self.block_resolutions:
			block = getattr(self, f'b{res}')
			x, img = block(x, img, self.curr_resolution, alpha=self.alpha, **block_kwargs)

		cmap = None
		if self.c_dim > 0:
			cmap = self.mapping(None, c)
		x = self.b4(x, img, cmap)
		return x

	def extra_repr(self):
		return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'
	
	def set_resolution(self, resolution):
		self.prev_resolution = self.curr_resolution
		self.curr_resolution = resolution
		self.alpha.copy_(torch.zeros([], device=self.alpha.device))
	
	def resolution_parameters(self) -> Iterator[Parameter]:
		return self.parameters()
#----------------------------------------------------------------------------
