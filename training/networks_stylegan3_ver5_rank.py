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
import scipy.signal
import scipy.optimize
from scipy.signal import gauss_spline
import torch

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops.siren_pytorch import SirenNetIPE, SirenNet
from torchvision.utils import save_image

DEBUG=False
# DEBUG=True

#----------------------------------------------------------------------------
def depthwise_demod_conv2d(
	x,
	w,
	groups		= 1,
	demodulate  = True,
	padding     = 0,
	stride		= 1,
	input_gain  = None,
	dilation	= 1,
):
	batch_size = int(x.shape[0])
	in_channels = int(x.shape[1])
	out_channels, _, kh, kw = w.shape
	misc.assert_shape(w, [out_channels, -1, kh, kw]) # [OIkk]
	misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]

	if input_gain is not None:
		input_gain = input_gain.expand(out_channels, 1) # [O]
		w = w * input_gain.unsqueeze(-1).unsqueeze(-1) # [OIkk]
	
	# Execute as one fused op using grouped convolution.
	x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=groups, dilation=dilation, stride=stride)
	return x

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

	def forward(self, x, gain=1):
		w = self.weight * self.weight_gain
		b = self.bias.to(x.dtype) if self.bias is not None else None
		flip_weight = (self.up == 1) # slightly faster
		if self.activation == 'linear':
			x = x.sin()
		x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

		act_gain = self.act_gain * gain
		act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
		x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
		return x

	def extra_repr(self):
		return ' '.join([
			f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
			f'up={self.up}, down={self.down}'])

@persistence.persistent_class
class Conv2dWrapper(torch.nn.Module):
	def __init__(self,
		activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
		up              = 1,            # Integer upsampling factor.
		down            = 1,            # Integer downsampling factor.
		resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
		conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
	):
		super().__init__()
		self.activation = activation
		self.up = up
		self.down = down
		self.conv_clamp = conv_clamp
		self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
		self.act_gain = bias_act.activation_funcs[activation].def_gain

	def forward(self, x, w, gain=1, padding=0, groups=1, up=1, down=1):
		_, I, kH, kW = w.shape
		weight_gain = 1 / np.sqrt(I * (kH * kW))
		w = w * weight_gain
		flip_weight = (self.up == 1) # slightly faster
		x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, flip_weight=flip_weight, padding=padding, groups=groups, up=up, down=down)
		return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DepthwiseSynthesisKernel(torch.nn.Module):
	def __init__(self,
		in_dim,
		out_dim,        # Number of output channels.
		in_sampling_rate, 
		halfwidth,
		cutoff,
		max_sampling_rate,
		init_size,
		down_factor,
		up_factor		= 1,
		filter_size 	= 4,
		is_rgb			= False,
		demod			= True,
		w0_init			= 8,
	):
		super().__init__()
		# self.w_dim = w_dim
		self.out_dim = out_dim
		self.in_dim = in_dim
		self.hidden_dim = min(out_dim, in_dim)
		self.filter_size = filter_size

		freqs = torch.randn([self.hidden_dim, 2])
		radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
		freqs /= radii * radii.square().exp().pow(0.25)
		freqs *= w0_init #* (max_sampling_rate // in_sampling_rate)
		phases = torch.rand([self.hidden_dim]) - 0.5
		self.phases = torch.nn.Parameter(phases)

		self.register_buffer('freqs', freqs)
		# self.register_buffer('phases', phases)
		self.register_buffer('hz_mean', torch.zeros([]))
		self.register_buffer('hz_std', torch.ones([self.out_dim,1,1,1]))
		self.register_buffer('vt_mean', torch.zeros([]))
		self.register_buffer('vt_std', torch.ones([self.out_dim,1,1,1]))

		self.hz_outdim = torch.nn.Parameter(torch.randn([self.in_dim, self.hidden_dim]))
		self.vt_outdim = torch.nn.Parameter(torch.randn([self.in_dim, self.hidden_dim]))

		# self.spectral_norm = [self.hidden_dim] * 2
		# self.hz_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))
		# self.vt_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))

		self.init_size = init_size
		self.out_size = init_size
		self.in_size = ((init_size+1)//2) * 4 * (max_sampling_rate // in_sampling_rate)
		self.sigma = 1 / init_size
		self.in_sampling_rate = in_sampling_rate
		self.max_sampling_rate = max_sampling_rate
		self.is_rgb = is_rgb
		self.density = 1
		self.demod = demod
		self.weight_gain = 1 / np.sqrt(self.in_dim * (self.out_size))
		self.ema_beta = 0.999
		self.cutoff = cutoff
		self.halfwidth = halfwidth*2
		num_taps = self.in_size - self.out_size * 2 + 2

		low_filter = design_lowpass_filter(numtaps=num_taps, cutoff=cutoff, width=halfwidth*2, fs=cutoff * 2**1.1)
		gauss_filter = np.arange(self.in_size) - (self.in_size-1) / 2
		self.register_buffer("low_filter", low_filter)
		self.register_buffer("gauss_filter", torch.from_numpy(gauss_filter).unsqueeze(0).to(dtype=self.low_filter.dtype))
		self.gauss_sigma = torch.nn.Parameter(torch.ones([]))

	def init_size_hyper(self, in_sampling_rate, cutoff, halfwidth):
		self.density = int(in_sampling_rate // self.in_sampling_rate)
		# self.in_size = ((self.init_size+1)//2) * 4 * self.density
		self.out_size = (self.density * self.init_size) + (self.density * self.init_size - 1) % 2
		num_taps = self.in_size - self.out_size * 2 + 2
		low_filter = design_lowpass_filter(numtaps=num_taps, cutoff=self.cutoff, width=self.halfwidth*2, fs=self.cutoff * 2**1.1)
		# low_filter = design_lowpass_filter(numtaps=num_taps, cutoff=cutoff, width=halfwidth*2, fs=cutoff * 2**1.1)
		self.register_buffer('low_filter', low_filter.to(self.low_filter.device))
		# self.low_filter.copy_(low_filter)
		
	def sample_grid(self, out_size, device):
		# x_grid = (torch.arange(out_size*2+1, device=device) - (out_size)) * (2 / (out_size*2+1))
		# y_grid = (torch.arange(out_size*2+1, device=device) - (out_size)) * (2 / (out_size*2+1))

		x_grid = (torch.arange(out_size, device=device) - ((out_size-1)/2)) * (2 / (out_size+1))
		y_grid = (torch.arange(out_size, device=device) - ((out_size-1)/2)) * (2 / (out_size+1))
		x_grid = self.freqs[:,0].unsqueeze(-1) * x_grid.unsqueeze(0) + self.phases.unsqueeze(-1)
		y_grid = self.freqs[:,1].unsqueeze(-1) * y_grid.unsqueeze(0) + self.phases.unsqueeze(-1)
		return x_grid, y_grid
	
	def sample_weight(self, grid, coeff):
		weight = torch.einsum('cg,ic->ig',torch.sin(grid * np.pi * 2) * np.exp(-1/(2*self.density**2)), coeff) / np.sqrt(self.hidden_dim)
		weight = weight - weight.mean()
		weight = weight * (weight.square().mean(dim=[0], keepdims=True) + 1e-8).rsqrt()
		gauss_sigma = 1 + self.gauss_sigma ** (2) / 5
		gauss_filter = (-self.gauss_filter ** 2 / (2 * gauss_sigma)).exp()
		weight = weight * gauss_filter 
		# weight = weight * self.gauss_filter / self.gauss_filter.max()
		weight = torch.nn.functional.conv1d(weight.unsqueeze(1), self.low_filter.view(1,1,-1)).squeeze(1)
		# weight = torch.nn.functional.interpolate(weight.unsqueeze(0), size=self.out_size, mode='linear')[0]
		return weight[:,::2]

	def forward(self, x, update_emas=True):
		
		device = x.device
		x_grid, y_grid = self.sample_grid(self.in_size, device)

		hz_weight = self.sample_weight(x_grid, self.hz_outdim)
		hz_weight = hz_weight.view(self.in_dim, 1, 1, self.out_size) 
		vt_weight = self.sample_weight(y_grid, self.vt_outdim)
		vt_weight = vt_weight.view(self.in_dim, 1, self.out_size, 1) 

		# hz_weight = hz_weight - hz_weight.mean(dim=[0])
		# hz_weight = hz_weight * (hz_weight.square().mean([0], keepdim=True) + 1e-8).rsqrt()

		# vt_weight = vt_weight - vt_weight.mean(dim=[0])
		# vt_weight = vt_weight * (vt_weight.square().mean([0], keepdim=True) + 1e-8).rsqrt()

		# if update_emas:
		# 	self.hz_mean.copy_(self.hz_mean.lerp_(hz_weight.mean(), self.ema_beta))
		# 	self.vt_mean.copy_(self.vt_mean.lerp_(vt_weight.mean(), self.ema_beta))

		# 	self.hz_std.copy_(self.hz_std.lerp_((hz_weight.square().mean([1,2,3], keepdim=True) + 1e-8).rsqrt(), self.ema_beta))
		# 	self.vt_std.copy_(self.vt_std.lerp_((vt_weight.square().mean([1,2,3], keepdim=True) + 1e-8).rsqrt(), self.ema_beta))

		# hz_weight = (hz_weight - self.hz_mean) 
		# hz_weight = hz_weight * self.hz_std
		# vt_weight = (vt_weight - vt_weight.mean())
		# vt_weight = vt_weight * self.vt_std

		# hz_weight *= self.weight_gain
		# vt_weight *= self.weight_gain

		# print("{:10s}: {:5f}, {:5f}".format('synthkernel', *torch.std_mean(weight))) if DEBUG else 1
		return hz_weight.type_as(x), vt_weight.type_as(x)



#----------------------------------------------------------------------------

@persistence.persistent_class
class NaiveSynthesisKernel(torch.nn.Module):
	def __init__(self,
		in_dim,
		out_dim,        # Number of output channels.
		in_sampling_rate, 
		max_sampling_rate,
		init_size,
		down_factor,
		up_factor		= 1,
		filter_size 	= 4,
		is_rgb			= False,
		demod			= True,
		w0_init			= 24,
	):
		super().__init__()
		# self.w_dim = w_dim
		self.out_dim = out_dim
		self.in_dim = in_dim
		self.hidden_dim = min(out_dim, in_dim)
		self.filter_size = filter_size

		freqs = torch.randn([self.hidden_dim, 2])
		radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
		freqs /= radii * radii.square().exp().pow(0.25)
		freqs *= 24
		phases = torch.rand([self.hidden_dim]) - 0.5

		self.register_buffer('freqs', freqs)
		self.register_buffer('phases', phases)
		self.register_buffer('spectral_reg', (-self.freqs.abs()*4).exp())

		self.hz_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))
		self.vt_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))

		# self.spectral_norm = [self.hidden_dim] * 2
		# self.hz_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))
		# self.vt_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))

		self.init_size = init_size
		self.out_size = init_size
		self.in_size = init_size 
		self.sigma = 1 / init_size
		self.in_sampling_rate = in_sampling_rate
		self.max_sampling_rate = max_sampling_rate
		self.is_rgb = is_rgb
		self.density = 1
		self.demod = demod
		self.weight_gain = 1 / np.sqrt(self.in_dim * (self.out_size))
		# self.test_weight = torch.nn.Parameter(torch.randn([self.out_dim, 1, self.out_size, self.out_size]) / np.sqrt(6))

		# gauss_size = int(self.max_sampling_rate / self.in_sampling_rate) * self.init_size
		gauss_size = self.out_size
		knots = np.arange(gauss_size) - gauss_size // 2
		gauss_filter = torch.from_numpy(gauss_spline(knots, 30))
		gauss_filter /= gauss_filter.sum()
		gauss_filter *= self.out_size
		self.register_buffer('gauss_filter', gauss_filter.view(1,1,gauss_size).cuda())
		# gauss_filter = (gauss_filter.view(1, -1) * gauss_filter.view(-1, 1)).sqrt()
		# gauss_filter /= gauss_filter.std() * 2

		# if hasattr(self, 'gauss_filter'):
		# 	self.register_buffer('gauss_filter', gauss_filter.to(self.gauss_filter.device))
		# else:

	
	def init_size_hyper(self, in_sampling_rate):
		if self.init_size != 1:
			self.density = int(in_sampling_rate // self.in_sampling_rate)
			self.out_size = self.density * self.init_size
			self.in_size = self.density * self.init_size
			self.sigma = 1 / self.density
			gauss_size = self.out_size
			knots = np.arange(gauss_size) - gauss_size // 2
			gauss_filter = torch.from_numpy(gauss_spline(knots, gauss_size))
			gauss_filter /= gauss_filter.sum()
			gauss_filter *= self.out_size
			self.register_buffer('gauss_filter', gauss_filter.view(1,1,gauss_size).cuda())
			self.weight_gain = np.sqrt(1 / ((self.out_size) * self.density * (self.out_dim)))
		
	def sample_grid(self, out_size, device):
		x_grid = (torch.arange(out_size*2+1, device=device) - (out_size)) * (2 / (out_size*2+1))
		y_grid = (torch.arange(out_size*2+1, device=device) - (out_size)) * (2 / (out_size*2+1))
		x_grid = self.freqs[:,0].unsqueeze(-1) * x_grid.unsqueeze(0) + self.phases.unsqueeze(-1)
		y_grid = self.freqs[:,1].unsqueeze(-1) * y_grid.unsqueeze(0) + self.phases.unsqueeze(-1)
		return x_grid, y_grid
	
	def sample_weight(self, grid, coeff, reg):
		out_size = grid.shape[1]//2
		coeff = coeff * reg.view(1, 1, -1)
		norm = reg.sum()
		weight = torch.nn.functional.interpolate(torch.sin(grid * np.pi * 2).unsqueeze(0), size=out_size, mode='linear', align_corners=False)[0]
		weight = torch.einsum('cg,oic->oig',weight, coeff) / norm.sqrt()
		return weight

	
	def forward(self, x, update_emas=True):
		
		if update_emas:
			self.spectral_reg.copy_(self.spectral_reg.lerp_(torch.ones_like(self.spectral_reg).to(x.device), 0.99))

		device = x.device
		x_grid, y_grid = self.sample_grid(self.out_size, device)

		hz_weight = self.sample_weight(x_grid, self.hz_outdim, self.spectral_reg[:,0])
		hz_weight = hz_weight.view(self.out_dim, self.in_dim, 1, self.out_size) 
		vt_weight = self.sample_weight(y_grid, self.vt_outdim, self.spectral_reg[:,1])
		vt_weight = vt_weight.view(self.out_dim, self.in_dim, self.out_size, 1) 

		if self.demod:
			hz_weight = (hz_weight - hz_weight.mean()) 
			hz_weight = hz_weight * (hz_weight.square().mean([1,2,3], keepdim=True) + 1e-8).rsqrt()
			vt_weight = (vt_weight - vt_weight.mean())
			vt_weight = vt_weight * (vt_weight.square().mean([1,2,3], keepdim=True) + 1e-8).rsqrt()

		# hz_weight *= self.gauss_filter.view(1, 1, 1, self.out_size).to(device)
		# vt_weight *= self.gauss_filter.view(1, 1, self.out_size, 1).to(device)

		hz_weight *= self.weight_gain
		vt_weight *= self.weight_gain

		# if self.out_size > 1:
		# 	weight = weight - weight.mean([1,2,3], keepdim=True)
		# 	weight = weight * weight.square().mean([1,2,3], keepdim=True).rsqrt()

		# print("{:10s}: {:5f}, {:5f}".format('synthkernel', *torch.std_mean(weight))) if DEBUG else 1
		return hz_weight.type_as(x), vt_weight.type_as(x)

	
	def extra_repr(self):
		return '\n'.join([
			f'density={self.density:d}, in_size={self.in_size:d}, out_size={list(self.out_size)},',])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisKernel(torch.nn.Module):
	def __init__(self,
		in_dim,
		out_dim,        # Number of output channels.
		in_sampling_rate, 
		max_sampling_rate,
		init_size,
		down_factor,
		up_factor		= 1,
		filter_size 	= 4,
		is_rgb			= False,
		demod			= False,
		w0_init			= 24,
	):
		super().__init__()
		# self.w_dim = w_dim
		self.out_dim = out_dim
		self.in_dim = in_dim
		self.hidden_dim = min(out_dim, in_dim)
		self.filter_size = filter_size
		self.hz_gen = SirenNet(
			dim_in = 1,
			dim_hidden = self.hidden_dim,
			dim_out = self.hidden_dim,
			num_layers = 2,
			w0_initial = w0_init,
			w0		   = w0_init,
			# final_activation=torch.nn.LeakyReLU(negative_slope=0.1)
		)
		self.hz_scale = torch.nn.Parameter(torch.randn([self.in_dim, 1])/10)
		self.hz_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))

		self.vt_gen = SirenNet(
			dim_in = 1,
			dim_hidden = self.hidden_dim,
			dim_out = self.hidden_dim,
			num_layers = 2,
			w0_initial = w0_init,
			w0		   = w0_init,
			# final_activation=torch.nn.LeakyReLU(negative_slope=0.1)
		)
		self.vt_scale = torch.nn.Parameter(torch.randn([self.in_dim, 1])/10)
		self.vt_outdim = torch.nn.Parameter(torch.randn([self.out_dim, self.in_dim, self.hidden_dim]))

		self.init_size = init_size
		self.out_size = init_size
		self.in_size = init_size 
		self.sigma = 1 / init_size
		self.in_sampling_rate = in_sampling_rate
		self.max_sampling_rate = max_sampling_rate
		self.is_rgb = is_rgb
		self.density = 1
		self.demod = demod
		self.weight_gain = 1 / np.sqrt(self.in_dim * (self.out_size))
		# self.test_weight = torch.nn.Parameter(torch.randn([self.out_dim, 1, self.out_size, self.out_size]) / np.sqrt(6))

		# gauss_size = int(self.max_sampling_rate / self.in_sampling_rate) * self.init_size
		gauss_size = self.out_size
		knots = np.arange(gauss_size) - gauss_size // 2
		gauss_filter = torch.from_numpy(gauss_spline(knots, 30))
		gauss_filter /= gauss_filter.sum()
		gauss_filter *= self.out_size
		self.register_buffer('gauss_filter', gauss_filter.view(1,1,gauss_size).cuda())
		# gauss_filter = (gauss_filter.view(1, -1) * gauss_filter.view(-1, 1)).sqrt()
		# gauss_filter /= gauss_filter.std() * 2

		# if hasattr(self, 'gauss_filter'):
		# 	self.register_buffer('gauss_filter', gauss_filter.to(self.gauss_filter.device))
		# else:

		self.hz_gen.freeze()
		self.vt_gen.freeze()
	
	def init_size_hyper(self, in_sampling_rate):
		if self.init_size != 1:
			self.density = int(in_sampling_rate // self.in_sampling_rate)
			self.out_size = self.density * self.init_size
			self.in_size = self.density * self.init_size
			self.sigma = 1 / self.density
			gauss_size = self.out_size
			knots = np.arange(gauss_size) - gauss_size // 2
			gauss_filter = torch.from_numpy(gauss_spline(knots, gauss_size))
			gauss_filter /= gauss_filter.sum()
			gauss_filter *= self.out_size
			self.register_buffer('gauss_filter', gauss_filter.view(1,1,gauss_size).cuda())
			self.weight_gain = 1 / np.sqrt((self.out_size) * self.density * (self.out_dim))
	
	def forward(self, x):
		device = x.device
		x_grid = (torch.arange(self.out_size, device=device) - (self.out_size - 1) / 2) * (2 / (self.out_size))
		y_grid = (torch.arange(self.out_size, device=device) - (self.out_size - 1) / 2) * (2 / (self.out_size))

		hz_weight = self.hz_gen(x_grid.view(-1, 1)).permute(1,0)
		# hz_weight = self.hz_scale.sigmoid() * hz_weight
		hz_weight = torch.einsum('cg,oic->oig',hz_weight,self.hz_outdim) / np.sqrt(self.hidden_dim)
		hz_weight = hz_weight.view(self.out_dim, self.in_dim, 1, self.out_size) 

		vt_weight = self.vt_gen(y_grid.view(-1, 1)).permute(1,0)
		# vt_weight = self.vt_scale.sigmoid() * vt_weight
		vt_weight = torch.einsum('cg,oic->oig',vt_weight,self.vt_outdim) / np.sqrt(self.hidden_dim)
		vt_weight = vt_weight.view(self.out_dim, self.in_dim, self.out_size, 1)

		if self.demod:
			hz_weight = hz_weight * (hz_weight.square().mean([1,2,3], keepdim=True) + 1e-8).rsqrt()
			vt_weight = vt_weight * (vt_weight.square().mean([1,2,3], keepdim=True) + 1e-8).rsqrt()

		# hz_weight *= self.gauss_filter.view(1, 1, 1, self.out_size).to(device)
		# vt_weight *= self.gauss_filter.view(1, 1, self.out_size, 1).to(device)

		hz_weight *= self.weight_gain
		vt_weight *= self.weight_gain

		# if self.out_size > 1:
		# 	weight = weight - weight.mean([1,2,3], keepdim=True)
		# 	weight = weight * weight.square().mean([1,2,3], keepdim=True).rsqrt()

		# print("{:10s}: {:5f}, {:5f}".format('synthkernel', *torch.std_mean(weight))) if DEBUG else 1
		return hz_weight.type_as(x), vt_weight.type_as(x)

	
	def extra_repr(self):
		return '\n'.join([
			f'density={self.density:d}, in_size={self.in_size:d}, out_size={list(self.out_size)},',])

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
		freq_channels,  # Number of frequency channels.
		channels,       # Number of output channels.
		size,           # Output spatial size: int or [width, height].
		sampling_rate,  # Output sampling rate.
		bandwidth,      # Output bandwidth.
	):
		super().__init__()
		self.w_dim = w_dim
		self.freq_channels = freq_channels
		self.channels = channels
		self.size = np.broadcast_to(np.asarray(size), [2])
		self.sampling_rate = sampling_rate
		self.init_sampling_rate = sampling_rate
		self.bandwidth = bandwidth
		self.density = 1

		# Draw random frequencies from uniform 2D disc.

		freqs = torch.randn([self.channels, 2])
		radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
		freqs /= radii * radii.square().exp().pow(0.25)
		freqs *= bandwidth
		phases = torch.rand([self.channels]) - 0.5

		# Setup parameters and buffers.
		self.weight = torch.nn.Parameter(torch.randn([self.channels, self.freq_channels]))
		self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1,0,0,0])
		self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
		self.register_buffer('freqs', freqs)
		self.register_buffer('phases', phases)
	
	def init_size_hyper(self, 
			size,
			sampling_rate,
			bandwidth,
		):
			self.size = np.broadcast_to(np.asarray(size), [2])
			self.sampling_rate = sampling_rate
			self.bandwidth = bandwidth
			self.density = self.sampling_rate / self.init_sampling_rate 

	def forward(self, w):

		# Introduce batch dimension and mask out unused bandwidth
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
		x = torch.sin(x * (np.pi * 2)) * np.exp(1 - 1/(self.density))
		x = x * amplitudes.unsqueeze(1).unsqueeze(2)

		# Apply trainable mapping.
		weight = self.weight / np.sqrt(self.freq_channels)
		x = x @ weight.t()

		# Ensure correct shape.
		x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
		misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
		return x * np.exp(1/2)

	def extra_repr(self):
		return '\n'.join([
			f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
			f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
	def __init__(self,
		w_dim,                          # Intermediate latent (W) dimensionality.
		is_torgb,                       # Is this the final ToRGB layer?
		is_critically_sampled,          # Does this layer use critical sampling?
		use_fp16,                       # Does this layer use FP16?

		# Input & output specifications.
		in_channels,                    # Number of input channels.
		out_channels,                   # Number of output channels.
		in_size,                        # Input spatial size: int or [width, height].
		out_size,                       # Output spatial size: int or [width, height].
		in_sampling_rate,               # Input sampling rate (s).
		out_sampling_rate,              # Output sampling rate (s).
		max_sampling_rate,
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
		self.is_critically_sampled = is_critically_sampled
		self.use_fp16 = use_fp16
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.in_size = np.broadcast_to(np.asarray(in_size), [2])
		self.out_size = np.broadcast_to(np.asarray(out_size), [2])
		self.in_sampling_rate = in_sampling_rate
		self.out_sampling_rate = out_sampling_rate
		self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
		self.in_cutoff = in_cutoff
		self.out_cutoff = out_cutoff
		self.in_half_width = in_half_width
		self.out_half_width = out_half_width
		self.conv_clamp = conv_clamp
		self.magnitude_ema_beta = magnitude_ema_beta
		self.filter_size = filter_size
		self.use_radial_filters = use_radial_filters
		self.is_torgb = is_torgb
		self.lrelu_upsampling = lrelu_upsampling

		# Setup parameters and buffers.
		self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
		self.lr_weight = torch.nn.Parameter(torch.randn([self.in_channels, self.in_channels, 1, 1]))
		self.style_weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, 1, 1]))
		self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
		self.register_buffer('magnitude_ema', torch.ones([]))
	
		self.weight_gen = DepthwiseSynthesisKernel(
			in_dim=self.in_channels,
			out_dim=self.in_channels,
			in_sampling_rate=in_sampling_rate,
			cutoff=self.out_cutoff,
			halfwidth=self.out_half_width,
			max_sampling_rate=max_sampling_rate,
			init_size=conv_kernel,
			down_factor=2,
			demod=True,
		)

		self.init_size_hyper(in_size, out_size, in_sampling_rate, out_sampling_rate, in_cutoff, out_cutoff, in_half_width, out_half_width)

	def init_size_hyper(self, 
			in_size,
			out_size,
			in_sampling_rate,
			out_sampling_rate,
			in_cutoff,
			out_cutoff,
			in_half_width,
			out_half_width,
			density=1
		):
			self.in_size = np.broadcast_to(np.asarray(in_size), [2])
			self.out_size = np.broadcast_to(np.asarray(out_size), [2])
			self.in_sampling_rate = in_sampling_rate
			self.out_sampling_rate = out_sampling_rate
			self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if self.is_torgb else self.lrelu_upsampling)
			self.in_cutoff = in_cutoff
			self.out_cutoff = out_cutoff
			self.in_half_width = in_half_width
			self.out_half_width = out_half_width
			self.density = density

			# Design upsampling filter.
			self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
			assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
			self.up_taps = self.filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
			up_filter = self.design_lowpass_filter(
				numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate)
			
			if hasattr(self, 'up_filter') and up_filter is not None:
				self.register_buffer('up_filter', up_filter.to(self.up_filter.device))
			else:
				self.register_buffer('up_filter', up_filter)

			# Design downsampling filter.
			self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
			assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
			self.down_taps = self.filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
			self.down_radial = self.use_radial_filters and not self.is_critically_sampled
			down_filter = self.design_lowpass_filter(
				numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial)

			if hasattr(self, 'down_filter') and down_filter is not None:
				self.register_buffer('down_filter', down_filter.to(self.down_filter.device))
			else:
				self.register_buffer('down_filter', down_filter)

			# TODO: Change here to weight_gen layer
			self.weight_gen.init_size_hyper(in_sampling_rate=self.in_sampling_rate, cutoff=self.out_cutoff, halfwidth=self.out_half_width)
			self.conv_kernel = 1 if self.is_torgb else self.weight_gen.out_size

			# Compute padding.
			pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
			pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
			pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
			pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
			pad_hi = pad_total - pad_lo
			self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

			# Compute sin padding
			pad_total = (self.in_size - 1) * 2 + 1
			pad_total -= (self.in_size) * 2
			pad_total += self.up_taps * 2 - 2
			pad_lo = (pad_total + 2) // 2
			pad_hi = pad_total - pad_lo
			self.sin_padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

	def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
		assert noise_mode in ['random', 'const', 'none'] # unused
		misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
		misc.assert_shape(w, [x.shape[0], self.w_dim])

		# Execute affine layer.
		styles = self.affine(w)
		if self.is_torgb:
			weight_gain = 1 / np.sqrt(self.in_channels)
			styles = styles * weight_gain
		
		# Execute demodulated depthwise conv2d.
		dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
		# Execute bias, filtered leaky ReLU, and clamping.

		if self.is_torgb is False:
			hz_weight, vt_weight = self.weight_gen(x, update_emas)
			lr_x = depthwise_demod_conv2d(x=x.to(dtype), w=self.lr_weight.to(dtype), groups=1) / np.sqrt(self.in_channels)
			lr_x = torch.nn.functional.pad(lr_x, [self.conv_kernel-1]*4)
			hz_x = depthwise_demod_conv2d(x=lr_x.to(dtype), w=hz_weight, groups=self.in_channels) 
			x = depthwise_demod_conv2d(x=hz_x.to(dtype), w=vt_weight, groups=self.in_channels) 

		# if self.is_torgb is False:
		# 	lr_x = filtered_lrelu.filtered_sin_ref(x=lr_x.to(torch.float32), fu=self.up_filter, fd=self.up_filter, padding=self.sin_padding,
		# 		up=2, down=2)

		# lr_x = depthwise_demod_conv2d(x=x.to(dtype), w=self.lr_weight.to(dtype), groups=1) / np.sqrt(self.in_channels)
		# lr_x = torch.nn.functional.pad(lr_x, [self.conv_kernel-1]*4)
		# hz_x = depthwise_demod_conv2d(x=lr_x.to(dtype), w=hz_weight) 
		# x = depthwise_demod_conv2d(x=hz_x.to(dtype), w=vt_weight) #/ np.sqrt(2) #* np.sqrt(max(self.weight_gen.density-1,1))

		if update_emas:
			with torch.autograd.profiler.record_function('update_magnitude_ema'):
				magnitude_cur = x.detach().to(torch.float32).square().mean()
				self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
		input_gain = self.magnitude_ema.rsqrt()

		x = modulated_conv2d(x=x.to(dtype), w=self.style_weight, s=styles,
			padding=0, demodulate=(not self.is_torgb), input_gain=input_gain)

		gain = 1 if self.is_torgb else np.sqrt(1/2)
		# gain = 1
		slope = 1 if self.is_torgb else 0.2
		x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
			up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

		# Ensure correct shape and dtype.
		misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
		assert x.dtype == dtype
		# return x 
		return x if self.is_torgb else 1.5 * x * (-x**2/10).exp()

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
		img_resolution,                 # Output image resolution.
		img_channels,                   # Number of color channels.
		max_img_resolution  = 256,
		channel_base        = 32768,    # Overall multiplier for the number of channels.
		channel_max         = 512,      # Maximum number of channels in any layer.
		num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
		num_critical        = 1,        # Number of critically sampled layers at the end.
		first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
		first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
		last_stopband_rel   = 2**0.2,   # Minimum stopband of the last layer, expressed relative to the cutoff.
		margin_size			= 20,
		output_scale        = 0.25,     # Scale factor for the output image.
		num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
		freq_channels       = 512,
		**layer_kwargs,                 # Arguments for SynthesisLayer.
	):
		super().__init__()
		self.w_dim = w_dim
		self.img_resolution = img_resolution
		self.num_ws = num_layers + 2
		self.img_channels = img_channels
		self.num_layers = num_layers
		self.num_critical = num_critical
		self.margin_size = margin_size
		self.output_scale = output_scale
		self.num_fp16_res = num_fp16_res
		self.last_stopband_rel = last_stopband_rel
		self.first_cutoff = first_cutoff
		self.first_stopband = first_stopband
		self.density = 1

		stopbands, cutoffs, sampling_rates, half_widths, sizes = self.compute_band(self.img_resolution)
		self.min_sampling_rates = sampling_rates

		# Compute channels based on maximum resolution 
		max_conf = self.compute_band(max_img_resolution)
		max_cutoffs = max_conf[1]
		max_sampling_rates = max_conf[2]

		channels = np.rint(np.minimum((channel_base / 2) / max_cutoffs, channel_max))
		channels[-1] = self.img_channels

		# Construct layers.
		self.input = SynthesisInput(
			w_dim=self.w_dim, channels=int(channels[0]), freq_channels=freq_channels, size=int(sizes[0]),
			sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
		self.layer_names = []
		for idx in range(self.num_layers + 1):
			prev = max(idx - 1, 0)
			is_torgb = (idx == self.num_layers)
			is_critically_sampled = (idx >= self.num_layers - self.num_critical)
			use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
			layer = SynthesisLayer(
				w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
				in_channels=int(channels[prev]), out_channels= int(channels[idx]),
				in_size=int(sizes[prev]), out_size=int(sizes[idx]),
				in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]), max_sampling_rate=int(max_sampling_rates[idx]),
				in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
				in_half_width=half_widths[prev], out_half_width=half_widths[idx],
				**layer_kwargs)
			name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
			setattr(self, name, layer)
			self.layer_names.append(name)
	
	def init_size_hyper(self, img_resolution):
		density = int(img_resolution // self.img_resolution)
		# Currently assume that we only target integer increasement
		if density == self.density:
			return

		self.density = density

		stopbands, cutoffs, sampling_rates, half_widths, sizes = self.compute_band(img_resolution)
		# sizes[0] += (density % 4 != 0)

		self.input.init_size_hyper(size=int(sizes[0]), sampling_rate=int(sampling_rates[0]), bandwidth=cutoffs[0])

		for idx in range(self.num_layers + 1):
			layer = self.layer_names[idx]
			prev = max(idx - 1, 0)
			getattr(self, layer).init_size_hyper(in_size=int(sizes[prev]), out_size=int(sizes[idx]), 
												 in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
												 in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
												 in_half_width=half_widths[prev], out_half_width=half_widths[idx], density=density)

	def compute_band(self, img_resolution):

		# Geometric progression of layer cutoffs and min. stopbands.
		last_cutoff = img_resolution // 2 # f_{c,N}
		last_stopband = last_cutoff * self.last_stopband_rel # f_{t,N}
		exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
		cutoffs = self.first_cutoff * (last_cutoff / self.first_cutoff) ** exponents # f_c[i]
		stopbands = self.first_stopband * (last_stopband / self.first_stopband) ** exponents # f_t[i]

		# Compute remaining layer parameters.
		sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, img_resolution)))) # s[i]
		half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
		margin_size = np.clip((np.rint(np.log2(last_cutoff // self.first_stopband))) * 4, a_min=0, a_max=self.margin_size)
		sizes = sampling_rates + margin_size * 2
		# if hasattr(self, 'min_sampling_rates'):
		# 	# For pixel shuffle
		# 	density = sampling_rates / self.min_sampling_rates
		# 	sizes -= sizes % density
		# 	assert ((sizes % density) == 0).all()

		sizes[-2:] = img_resolution

		return stopbands, cutoffs, sampling_rates, half_widths, sizes

	def forward(self, ws, img_resolution, **layer_kwargs):
		misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
		ws = ws.to(torch.float32).unbind(dim=1)

		# Compute valid sampling rate / bandwidht / size from init_size
		if img_resolution is None:
			img_resolution = self.img_resolution * self.density

		self.init_size_hyper(img_resolution)

		# Execute layers.
		x = self.input(ws[0])
		for name, w in zip(self.layer_names, ws[1:]):
			x = getattr(self, name)(x, w, **layer_kwargs)
			pad = (x.shape[-1] - getattr(self, name).out_sampling_rate)//2
			sample_image = x[0,:3,pad:-pad,pad:-pad] if pad !=0 else x[0,:1]
			save_image(sample_image * 10, f'V{name}_{img_resolution}.png')
			print("{:10s}: {:5f}, {:5f}".format(name, *torch.std_mean(x))) if DEBUG else 1
		if self.output_scale != 1:
			x = x * self.output_scale

		# Ensure correct shape and dtype.
		misc.assert_shape(x, [None, self.img_channels, self.img_resolution * self.density, self.img_resolution * self.density])
		x = x.to(torch.float32)

		print("{:10s}: {:5f}, {:5f}".format("G_out", *torch.std_mean(x))) if DEBUG else 1
		return x

	def extra_repr(self):
		return '\n'.join([
			f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
			f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
			f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
			f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
	def __init__(self,
		z_dim,                      # Input latent (Z) dimensionality.
		c_dim,                      # Conditioning label (C) dimensionality.
		w_dim,                      # Intermediate latent (W) dimensionality.
		img_resolution,             # Output resolution.
		img_channels,               # Number of output color channels.
		mapping_kwargs      = {},   # Arguments for MappingNetwork.
		**synthesis_kwargs,         # Arguments for SynthesisNetwork.
	):
		super().__init__()
		self.z_dim = z_dim
		self.c_dim = c_dim
		self.w_dim = w_dim
		self.img_channels = img_channels
		self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
		self.num_ws = self.synthesis.num_ws
		self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

	def forward(self, z, c, img_resolution=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
		ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
		img = self.synthesis(ws, img_resolution, update_emas=update_emas, **synthesis_kwargs)
		return img
	
	def init_size_hyper(self, resolution):
		self.synthesis.init_size_hyper(resolution)
	
#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
	def __init__(self,
		in_channels,                        # Number of input channels, 0 = first block.
		out_channels,                       # Number of output channels.
		in_size,                         	# Size of input
		out_size,							# Size of output
		in_sampling_rate,
		out_sampling_rate,
		max_sampling_rate,
		in_cutoff,
		out_cutoff,
		in_half_width,
		out_half_width,
		img_channels,                       # Number of input color channels.
		first_layer_idx,                    # Index of the first layer.
		conv_kernel			= 3,
		architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
		activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
		conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
		use_fp16            = False,        # Use FP16 for this block?
		fp16_channels_last  = False,        # Use channels-last memory format with FP16?
		freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
		filter_size			= 3
	):
		assert architecture in ['orig', 'skip', 'resnet']
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.img_channels = img_channels
		self.first_layer_idx = first_layer_idx
		self.architecture = architecture
		self.use_fp16 = use_fp16
		self.channels_last = (use_fp16 and fp16_channels_last)
		self.num_layers = 0
		self.conv_clamp = conv_clamp
		self.filter_size = filter_size

		if first_layer_idx == 0 or architecture == 'skip':
			self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1,
				conv_clamp=conv_clamp, channels_last=self.channels_last)
		
		self.conv_wrap = Conv2dWrapper(conv_clamp=conv_clamp)
	
		self.conv0_gen = DepthwiseSynthesisKernel(
			in_dim=self.in_channels,
			out_dim=self.in_channels,
			in_sampling_rate=in_sampling_rate,
			max_sampling_rate=max_sampling_rate,
			init_size=conv_kernel,
			down_factor=2,
			cutoff=out_cutoff,
			halfwidth=out_half_width
		)

		self.lr_weight0 = torch.nn.Parameter(torch.randn(in_channels ,in_channels, 1, 1) / np.sqrt(in_channels))
		self.conv0_point = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=True,
			conv_clamp=conv_clamp, channels_last=self.channels_last, activation='lrelu')
		
		self.conv1_gen = DepthwiseSynthesisKernel(
			in_dim=self.out_channels,
			out_dim=self.out_channels,
			in_sampling_rate=in_sampling_rate,
			max_sampling_rate=max_sampling_rate,
			init_size=conv_kernel,
			down_factor=2,
			cutoff=out_cutoff,
			halfwidth=out_half_width
		)

		self.lr_weight1 = torch.nn.Parameter(torch.randn(out_channels, out_channels, 1, 1) / np.sqrt(out_channels//2))
		self.conv1_point = Conv2dLayer(out_channels, out_channels, kernel_size=1, bias=True,
			conv_clamp=conv_clamp, channels_last=self.channels_last, activation='lrelu')
		
		if architecture == 'resnet':
			self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False,
				channels_last=self.channels_last)
		
		self.init_size_hyper(
			in_size=in_size,
			out_size=out_size,
			in_sampling_rate=in_sampling_rate,
			out_sampling_rate=out_sampling_rate,
			in_cutoff=in_cutoff,
			out_cutoff=out_cutoff,
			in_half_width=in_half_width,
			out_half_width=out_half_width,
			density=1,
		)

	def init_size_hyper(self, 
			in_size,
			out_size,
			in_sampling_rate,
			out_sampling_rate,
			in_cutoff,
			out_cutoff,
			in_half_width,
			out_half_width,
			density=1
		):
			self.in_size = np.broadcast_to(np.asarray(in_size), [2])
			self.out_size = np.broadcast_to(np.asarray(out_size), [2])
			self.in_sampling_rate = in_sampling_rate
			self.out_sampling_rate = out_sampling_rate
			self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * 2
			self.in_cutoff = in_cutoff
			self.out_cutoff = out_cutoff
			self.in_half_width = in_half_width
			self.out_half_width = out_half_width

			# self.down_factor = 4 if (self.out_size[0] // self.in_size[0] > 1) else 1
			# self.up_factor = (self.down_factor + 1) // 2
			self.up_factor = self.tmp_sampling_rate // self.in_sampling_rate
			self.down_factor = min(self.tmp_sampling_rate // self.out_sampling_rate, 3)

			self.density = density
			self.conv0_gen.init_size_hyper(in_sampling_rate=in_sampling_rate, cutoff=self.out_cutoff, halfwidth=self.out_half_width)
			self.conv1_gen.init_size_hyper(in_sampling_rate=in_sampling_rate, cutoff=self.out_cutoff, halfwidth=self.out_half_width)
			self.conv_kernel = self.conv0_gen.out_size

			pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
			pad_total -= (self.in_size) * self.up_factor # Input size after upsampling.
			pad_total += 4 * 1 - 2 # Size reduction caused by the filters.
			pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
			pad_hi = pad_total - pad_lo
			self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]


	def forward(self, x, img, update_emas=False, force_fp32=False):
		if (x if x is not None else img).device.type != 'cuda':
			force_fp32 = True
		dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
		memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

		# Input.
		if x is not None:
			misc.assert_shape(x, [None, self.in_channels, *self.in_size])
			x = x.to(dtype=dtype, memory_format=memory_format)

		# FromRGB.
		if self.first_layer_idx == 0 or self.architecture == 'skip':
			misc.assert_shape(img, [None, self.img_channels, *self.in_size])
			img = img.to(dtype=dtype, memory_format=memory_format)
			y = self.fromrgb(img)
			x = x + y if x is not None else y
			# img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

		# Main layers.
		if self.architecture == 'resnet':
			y = self.skip(x, gain=np.sqrt(0.5))
			hz_w0, vt_w0 = self.conv0_gen(x, update_emas)
			
			lr_x0 = self.conv_wrap(x, self.lr_weight0) / np.sqrt(self.in_channels)
			lr_x0 = torch.nn.functional.pad(lr_x0, pad=[(self.conv_kernel-1)//2, (self.conv_kernel-1)//2, (self.conv_kernel-1)//2, (self.conv_kernel-1)//2])
			hz_x0 = self.conv_wrap(lr_x0, hz_w0, groups=int(self.in_channels))
			vt_x0 = self.conv_wrap(hz_x0, vt_w0, groups=int(self.in_channels))
			x = self.conv0_point(vt_x0)


			hz_w1, vt_w1 = self.conv1_gen(x, update_emas)
			lr_x1 = self.conv_wrap(x, self.lr_weight1) / np.sqrt(self.out_channels)
			lr_x1 = torch.nn.functional.pad(lr_x1, pad=[(self.conv_kernel-1)//2, (self.conv_kernel-1)//2, (self.conv_kernel-1)//2, (self.conv_kernel-1)//2])
			hz_x1 = self.conv_wrap(lr_x1, hz_w1, groups=int(self.out_channels))
			vt_x1 = self.conv_wrap(hz_x1, vt_w1, groups=int(self.out_channels))
			vt_x1 = y.add_(vt_x1)

			self.conv1_point.padding = self.padding
			self.conv1_point.up = self.up_factor
			self.conv1_point.down = self.down_factor
			x = self.conv1_point(vt_x1)
		else:
			x = self.conv0(x)
			x = self.conv1(x)

		assert x.dtype == dtype

		return 1.5 * x * (-x**2/10).exp() , img
		# return x, img

	def extra_repr(self):
		return f'resolution={self.in_size[0]:d}, architecture={self.architecture:s}'

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
		# in_size = x.shape[-1] // 2
		# cx = x[...,in_size-2:in_size+2, in_size-2:in_size+2]
		cx = x
		misc.assert_shape(cx, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
		_ = force_fp32 # unused
		dtype = torch.float32
		memory_format = torch.contiguous_format

		# FromRGB.
		cx = cx.to(dtype=dtype, memory_format=memory_format)

		# Main layers.
		if self.mbstd is not None:
			cx = self.mbstd(cx)
		cx = self.conv(cx)
		cx = self.fc(cx.flatten(1))
		cx = self.out(cx)

		# Conditioning.
		if self.cmap_dim > 0:
			misc.assert_shape(cmap, [None, self.cmap_dim])
			cx = (cx * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

		assert cx.dtype == dtype
		return cx

	def extra_repr(self):
		return f'in_channels={self.in_channels:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
	def __init__(self,
		c_dim,                          # Conditioning label (C) dimensionality.
		img_resolution,                 # Input resolution.
		img_channels,                   # Number of input color channels.
		max_img_resolution	= 512,
		out_resolution      = 8,        # Output resolution.
		architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
		channel_base        = 32768,    # Overall multiplier for the number of channels.
		channel_max         = 512,      # Maximum number of channels in any layer.
		num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
		conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
		cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
		last_cutoff         = 2**1,     # Cutoff frequency of the first layer
		last_stopband       = 2**2.1,   # Minimum stopband of the first layer
		first_stopband_rel   = 2**-0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff
		num_layers          = 6,        # Number of layers
		margin_size         = 10,
		block_kwargs        = {},       # Arguments for DiscriminatorBlock.
		mapping_kwargs      = {},       # Arguments for MappingNetwork.
		epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
	):
		super().__init__()
		self.c_dim = c_dim
		self.img_resolution = img_resolution
		self.out_resolution = out_resolution
		self.num_layers = num_layers
		self.margin_size = margin_size
		self.last_cutoff =  last_cutoff
		self.last_stopband = last_stopband
		self.first_stopband_rel = first_stopband_rel

		stopbands, cutoffs, sampling_rates, half_widths, sizes = self.compute_band(self.img_resolution)

		self.stopbands = stopbands
		self.cutoffs = cutoffs
		self.min_sampling_rates = sampling_rates

		self.sizes = sizes
		self.img_channels = img_channels
		self.density = 1
		
		max_conf = self.compute_band(max_img_resolution)
		max_cutoff = max_conf[1]
		max_sampling_rates = max_conf[2]

		channels = np.exp2(np.ceil(np.log2(np.minimum(channel_base // (max_cutoff*4) , channel_max)))).astype(np.int)
		# channels = [int(min(np.rint(channel_base // 2 / cut), channel_max)) for cut in cutoffs]
		# channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}

		fp16_cutoff = 8

		if cmap_dim is None:
			cmap_dim = channels[-1]
		if c_dim == 0:
			cmap_dim = 0

		common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
		self.layer_name = []
		for idx in range(1, self.num_layers+1):
			prev = max(idx - 1, 0)
			in_channels = channels[prev] if prev < idx else img_channels
			out_channels = channels[idx]
			use_fp16 = (cutoffs[idx] >= fp16_cutoff)
			block = DiscriminatorBlock(in_channels, out_channels,
				first_layer_idx=idx-1, use_fp16=use_fp16, 
				in_size=int(sizes[prev]), out_size=int(sizes[idx]),
				in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]), max_sampling_rate=int(max_sampling_rates[prev]),
				in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
				in_half_width=half_widths[prev], out_half_width=half_widths[idx],
				**block_kwargs, **common_kwargs)
			setattr(self, f'b{idx}', block)
			self.layer_name.append(f'b{idx}')
		if c_dim > 0:
			self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
		self.b_ep = DiscriminatorEpilogue(channels[-1], cmap_dim=cmap_dim, resolution=out_resolution, **epilogue_kwargs, **common_kwargs)

	def compute_band(self, img_resolution):

		# Geometric progression of layer cutoffs and min. stopbands.
		first_cutoff = img_resolution // 2
		first_stopband = first_cutoff * self.first_stopband_rel
		exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers), 1)
		cutoffs = first_cutoff * (self.last_cutoff / first_cutoff) ** exponents
		stopbands = first_stopband * (self.last_stopband / first_stopband) ** exponents

		# Compute remaining layer parameters
		sampling_rates = np.exp2(np.ceil(np.log2(np.maximum(stopbands * 2, self.out_resolution))))
		half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs
		margin_size = np.clip((np.rint(np.log2(first_cutoff // self.last_cutoff)) - 2) * 3, a_min=0, a_max=self.margin_size)
		# sizes = sampling_rates + margin_size * 2
		sizes = sampling_rates

		# if hasattr(self, 'min_sampling_rates'):
		# 	# For pixel shuffle
		# 	density = sampling_rates / self.min_sampling_rates
		# 	sizes -= sizes % density
		# 	assert ((sizes % density) == 0).all()

		sizes[-1] = self.out_resolution

		return stopbands, cutoffs, sampling_rates, half_widths, sizes
	
	def init_size_hyper(self, img_resolution):
		density = int(img_resolution // self.img_resolution)
		if density == self.density:
			return 
		
		self.density = density
		stopbands, cutoffs, sampling_rates, half_widths, sizes = self.compute_band(img_resolution)
		self.sizes = sizes

		for idx in range(1, self.num_layers + 1):
			layer = self.layer_name[idx-1]
			prev = max(idx - 1, 0)
			getattr(self, layer).init_size_hyper(in_size=int(sizes[prev]), out_size=int(sizes[idx]), 
												 in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
												 in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
												 in_half_width=half_widths[prev], out_half_width=half_widths[idx], density=density)

	def forward(self, img, c, img_resolution=None, update_emas=False, **block_kwargs):
		if img_resolution is None:
			img_resolution = self.img_resolution * self.density
		
		self.init_size_hyper(img_resolution)
		_ = update_emas # unused
		x = None
		pad = (self.sizes[0] - img_resolution) // 2
		img = torch.nn.functional.pad(img, [int(pad)] * 4)
		for layer_name in self.layer_name:
			block = getattr(self, layer_name)
			x, img = block(x, img, update_emas=update_emas, **block_kwargs)
			print("{:10s}: {:5f}, {:5f}".format(layer_name, *torch.std_mean(x))) if DEBUG else 1

		cmap = None
		if self.c_dim > 0:
			cmap = self.mapping(None, c)

		print("{:10s}: {:5f}, {:5f}".format("logits", *torch.std_mean(x))) if DEBUG else 1
		x = self.b_ep(x, img, cmap)
		return x

	def extra_repr(self):
		return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
