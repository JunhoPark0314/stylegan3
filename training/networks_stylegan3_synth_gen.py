# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generator architecture from the paper
"Alias-Free Generative Adversarial Networks"."""

from collections import defaultdict
import numpy as np
import scipy.signal
import scipy.optimize
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act
from torch_utils.ops import upfirdn2d
from torchvision.utils import save_image

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

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
	def __init__(self,
		w_dim,          # Intermediate latent (W) dimensionality.
		channels,       # Number of output channels.
		size,           # Output spatial size: int or [width, height].
		sampling_rate,  # Output sampling rate.
		bandwidth,      # Output bandwidth.
		trainable_phase,
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
		if trainable_phase:
			self.phases = torch.nn.Parameter(phases)
		else:
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
		self.in_size = np.broadcast_to(np.asarray(in_size), [2])
		self.out_size = np.broadcast_to(np.asarray(out_size), [2])
		self.in_sampling_rate = in_sampling_rate
		self.out_sampling_rate = out_sampling_rate
		self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
		self.in_cutoff = in_cutoff
		self.out_cutoff = out_cutoff
		self.in_half_width = in_half_width
		self.out_half_width = out_half_width
		self.conv_kernel = 1 if is_torgb else conv_kernel
		self.conv_clamp = conv_clamp
		self.magnitude_ema_beta = magnitude_ema_beta

		# Setup parameters and buffers.
		self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
		self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
		# self.weight_gen = SynthesisGroupKernel(in_channels=self.in_channels, out_channels=self.out_channels, sampling_rate=self.in_sampling_rate)
		self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
		self.register_buffer('magnitude_ema', torch.ones([]))

		# Design upsampling filter.
		self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
		assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
		self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
		self.register_buffer('up_filter', self.design_lowpass_filter(
			numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

		# Design downsampling filter.
		self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
		assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
		self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
		self.down_radial = use_radial_filters and not self.is_critically_sampled
		self.register_buffer('down_filter', self.design_lowpass_filter(
			numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

		# Compute padding.
		pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
		pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
		pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
		pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
		pad_hi = pad_total - pad_lo
		self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]
		self.gamma = torch.nn.Parameter(torch.ones([self.out_channels]))

	def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
		assert noise_mode in ['random', 'const', 'none'] # unused
		misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
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
		# weight = self.weight_gen(device = x.device, ks=1 if self.is_torgb else 3).to(dtype)
		weight = self.weight
		x = modulated_conv2d(x=x.to(dtype), w=weight, s=styles,
			padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

		out_gain = x.square().mean(dim=[2,3]).rsqrt().clip(min=0.2, max=5)
		x = x * out_gain.unsqueeze(-1).unsqueeze(-1) * self.gamma.to(dtype).view(1, -1, 1, 1)

		# Execute bias, filtered leaky ReLU, and clamping.
		gain = 1 if self.is_torgb else np.sqrt(2)
		slope = 1 if self.is_torgb else 0.2
		x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
			up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

		# Ensure correct shape and dtype.
		misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
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
class ToRGBLayer(torch.nn.Module):
	def __init__(self, 
		in_channels, 
		out_channels, 
		w_dim, 
		low_cutoff, 
		in_size,
		sampling_rate, 
		img_resolution,
		kernel_size=1, 
		conv_clamp=None, 
		channels_last=False
	):

		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.w_dim = w_dim
		self.conv_clamp = conv_clamp
		self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
		memory_format = torch.channels_last if channels_last else torch.contiguous_format
		# self.weight_gen = SynthesisGroupKernel(in_channels=in_channels, out_channels=out_channels, sampling_rate=sampling_rate)
		self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
		self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
		self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
		self.up_factor = img_resolution // sampling_rate
		self.img_resolution = img_resolution
		self.in_size = in_size
		self.low_cutoff = low_cutoff
		self.sampling_rate = sampling_rate
		if low_cutoff != 0:
			high_pass_filter = self.design_lowpass_filter(numtaps=6*2+1, cutoff=low_cutoff * 0.25, width=low_cutoff * 2, fs=in_size, pass_zero=False)
			# high_pass_filter = self.design_lowpass_filter(numtaps=6*2+1, cutoff=low_cutoff * 0.25, width=low_cutoff * 0.25 * (np.sqrt(2) - 1), fs=in_size, pass_zero=False)
			self.register_buffer("pass_filter", high_pass_filter)
		if self.up_factor != 1:
			width = (sampling_rate - sampling_rate * (2 ** -0.9))
			up_filter = self.design_lowpass_filter(numtaps=6*self.up_factor+1, cutoff=sampling_rate * (2 ** -0.9) , width=width, fs=self.img_resolution)
			self.register_buffer("up_filter", up_filter)
		self.pad_size = (sampling_rate - in_size) * self.up_factor//2
	
	@staticmethod
	def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False, pass_zero=True):
		assert numtaps >= 1

		# Identity filter.
		if numtaps == 1:
			return None

		# Separable Kaiser low-pass filter.
		if not radial:
			f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs, pass_zero=pass_zero)
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

	def forward(self, x, w, fused_modconv=True):
		styles = self.affine(w) * self.weight_gain
		# weight = self.weight_gen(device=x.device, ks=1)
		weight = self.weight
		x = modulated_conv2d(x=x, w=weight, s=styles, demodulate=False, padding=0)
		x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
		# High pass filter	
		# if hasattr(self, "pass_filter"):
		# 	x = upfirdn2d.filter2d(x=x, f=self.pass_filter)
		if self.up_factor != 1:
			x = upfirdn2d.upsample2d(x=x, f=self.up_filter, up=self.up_factor) 
			x = torch.nn.functional.pad(x, [self.pad_size] * 4)

		misc.assert_shape(x, [None, self.out_channels, self.img_resolution, self.img_resolution])
		return x

	def extra_repr(self):
		return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
	def __init__(self,
		w_dim,                          # Intermediate latent (W) dimensionality.
		img_resolution,                 # Output image resolution.
		img_channels,                   # Number of color channels.
		channel_base        = 32768,    # Overall multiplier for the number of channels.
		channel_max         = 512,      # Maximum number of channels in any layer.
		num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
		num_critical        = 2,        # Number of critically sampled layers at the end.
		first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
		first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
		last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
		margin_size         = 10,       # Number of additional pixels outside the image.
		output_scale        = 0.25,     # Scale factor for the output image.
		num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
		trainable_phase     = False,
		**layer_kwargs,                 # Arguments for SynthesisLayer.
	):
		super().__init__()
		self.w_dim = w_dim
		self.num_ws = num_layers + 2
		self.img_resolution = img_resolution
		self.img_channels = img_channels
		self.num_layers = num_layers
		self.num_critical = num_critical
		self.margin_size = margin_size
		self.output_scale = output_scale
		self.num_fp16_res = num_fp16_res

		# Geometric progression of layer cutoffs and min. stopbands.
		last_cutoff = self.img_resolution / 2 # f_{c,N}
		last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
		exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
		cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
		stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

		# Compute remaining layer parameters.
		sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
		half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
		sizes = sampling_rates + self.margin_size * 2
		sizes[-2:] = self.img_resolution
		channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
		channels[-1] = self.img_channels

		# Construct layers.
		self.input = SynthesisInput(
			w_dim=self.w_dim, channels=int(channels[0]), size=int(sizes[0]),
			sampling_rate=sampling_rates[0], bandwidth=cutoffs[0], trainable_phase=trainable_phase)
		self.layer_names = []
		self.num_trgb = int(np.log2(self.img_resolution))
		self.to_rgb_layers = defaultdict(dict)
		self.min_channels = int(channels[-2])
		low_cutoff = 0
		for idx in range(self.num_layers+1):
			prev = max(idx - 1, 0)
			is_torgb = (idx == self.num_layers)
			is_critically_sampled = (idx >= self.num_layers - self.num_critical)
			use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)
			in_size = int(sizes[idx]) if is_torgb else int(sizes[prev]) 
			out_size = int(sizes[idx])
			layer = SynthesisLayer(
				w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
				in_channels=int(channels[prev]), out_channels= int(channels[idx]),
				in_size=in_size, out_size=out_size,
				in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
				in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
				in_half_width=half_widths[prev], out_half_width=half_widths[idx],
				**layer_kwargs)
			name = f'L{idx}_{layer.out_size[0]}_{layer.out_channels}'
			setattr(self, name, layer)
			self.layer_names.append(name)
			
			if (sizes[idx] < self.img_resolution) and (sizes[idx] != sizes[idx+1]):
				self.to_rgb_layers[name] = {
					"name": f"ToRGB_{int(sampling_rates[idx])}",
					"in_channels" : int(channels[idx]),
					"out_channels" : int(self.min_channels),
					"in_size" : int(sizes[idx]),
					"sampling_rate": int(sampling_rates[idx]),
					"low_cutoff" : low_cutoff,
				}
				low_cutoff = sampling_rates[idx] // 2
		self.to_rgb_layers[self.layer_names[-2]] = {
			"name": f"ToRGB_{self.img_resolution}",
			"in_channels" : int(self.min_channels),
			"out_channels" : int(self.min_channels),
			"in_size": int(sizes[idx-1]),
			"sampling_rate": self.img_resolution,
			"low_cutoff" : low_cutoff,
		}
		
		for k, v in self.to_rgb_layers.items():
			layer = ToRGBLayer(in_channels=v["in_channels"], out_channels=v["out_channels"], w_dim=self.w_dim, 
						low_cutoff = v["low_cutoff"], img_resolution=img_resolution,
						sampling_rate=v["sampling_rate"], in_size=v["in_size"], conv_clamp=256)
			name = v["name"]
			setattr(self, name, layer)

	def forward(self, ws, **layer_kwargs):
		misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
		ws = ws.to(torch.float32).unbind(dim=1)
		to_rgb_out = torch.zeros([ws[0].shape[0], self.min_channels, self.img_resolution, self.img_resolution], device=ws[0].device)

		# Execute layers.
		x = self.input(ws[0])
		for name, w in zip(self.layer_names[:-1], ws[1:-1]):
			x = getattr(self, name)(x, w, **layer_kwargs)
			if name in self.to_rgb_layers:
				trgb = getattr(self, self.to_rgb_layers[name]["name"])(x, w)
				to_rgb_out = to_rgb_out + trgb
		
		x = getattr(self, self.layer_names[-1])(to_rgb_out / np.sqrt(self.num_trgb), ws[-1], **layer_kwargs)
		
		if self.output_scale != 1:
			x = x * self.output_scale

		# Ensure correct shape and dtype.
		misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
		x = x.to(torch.float32)
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
		self.img_resolution = img_resolution
		self.img_channels = img_channels
		self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
		self.num_ws = self.synthesis.num_ws
		self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

	def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
		ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
		img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
		return img

#----------------------------------------------------------------------------

class SynthesisGroupKernel(torch.nn.Module):
	def __init__(self,
		in_channels,
		out_channels,
		cutoff = None,
		sampling_rate = 16,
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.sampling_rate = sampling_rate
		self.cutoff = cutoff * 2 if cutoff is not None else sampling_rate
		self.bandwidth = self.sampling_rate * (2 ** 0.3) #* (2 ** -0.9)
		self.freq_dim = np.clip(self.sampling_rate * 4, a_min=512, a_max=4096)


		# Draw random frequencies from uniform 2D disc.
		freqs = torch.randn([self.in_channels, self.freq_dim, 2])
		radii = freqs.square().sum(dim=-1, keepdim=True).sqrt()
		dist = torch.randn([self.in_channels, self.freq_dim, 1]).sin()
		dist = torch.sort(dist, dim=1)[0]
		# freqs /= radii * radii.square().exp().pow(0.25)
		freqs /= radii * dist
		freqs *= self.bandwidth
		phases = torch.rand([self.in_channels, self.freq_dim]) - 0.5

		# self.register_buffer("dist", dist.squeeze(-1).abs().clip(min=0.3, max=0.6))
		self.register_buffer("freqs", freqs)
		# self.register_buffer("phases", phases)
		self.phases = torch.nn.Parameter(phases)

		self.freq_weight = torch.nn.Parameter(torch.randn([in_channels, self.freq_dim]))
		self.freq_out = torch.nn.Parameter(torch.randn([out_channels, self.freq_dim]))
		self.weight_bias = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, 1]))

		# self.gain = lambda x: np.sqrt(1 / (in_channels * (x ** 2)))
		self.gain = 1


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

		kernel = kernel.squeeze(0) * self.gain

		return kernel

	def freq_parameters(self):
		params = []
		for k, v in self.named_parameters():
			if k in self.freq_param:
				params.append(v)
		return params

#----------------------------------------------------------------------------