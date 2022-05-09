import os
import copy
from random import choices
import numpy as np
from scipy.fft import fft
import torch
import torch.fft
import scipy.ndimage
import matplotlib.pyplot as plt
import click
import tqdm
import dnnlib
import json
import re
from torch.nn.utils import prune

import legacy
from metrics.metric_utils import get_feature_detector
from training import dataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from training.networks_stylegan2 import Conv2dLayer

#----------------------------------------------------------------------------
# Setup an iterator for streaming images, in uint8 NCHW format, based on the
# respective command line options.

def stream_source_images(source, num, seed, device, data_loader_kwargs=None, batch_size=64): # => num_images, image_size, image_iter
	ext = source.split('.')[-1].lower()
	if data_loader_kwargs is None:
		data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

	if ext == 'pkl':
		if num is None:
			raise click.ClickException('--num is required when --source points to network pickle')
		with dnnlib.util.open_url(source) as f:
			G = legacy.load_network_pkl(f)['G_ema'].to(device)
		def generate_image(seed):
			rnd = np.random.RandomState(seed)
			z = torch.from_numpy(rnd.randn(1, G.z_dim)).to(device)
			c = torch.zeros([1, G.c_dim], device=device)
			if G.c_dim > 0:
				c[:, rnd.randint(G.c_dim)] = 1
			return (G(z=z, c=c) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
		_ = generate_image(seed) # warm up
		image_iter = (generate_image(seed + idx) for idx in range(num))
		return num, G.img_resolution, image_iter

	elif ext == 'zip' or os.path.isdir(source):
		dataset_obj = dataset.ImageFolderDataset(path=source, max_size=num, random_seed=seed)
		if num is not None and num != len(dataset_obj):
			raise click.ClickException(f'--source contains fewer than {num} images')
		data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size, **data_loader_kwargs)
		image_iter = (image.to(device) for image, _label in data_loader)
		return len(dataset_obj), dataset_obj.resolution, image_iter

	else:
		raise click.ClickException('--source must point to network pickle, dataset zip, or directory')

#----------------------------------------------------------------------------
# Load average power spectrum from the specified .npz file and construct
# the corresponding heatmap for visualization.

def construct_heatmap(npz_file, smooth, key='spectrum', power=True):
	npz_data = np.load(npz_file)
	npz_key  = [f.split('.')[0] for f in npz_data._files]
	hmap_dict = {}
	image_size_dict = {}
	for i_key in npz_key:
		spectrum = npz_data[i_key] + 1e-6
		image_size = spectrum.shape[-1]
		hmap = np.log10(spectrum) * 10 # dB
		hmap = np.fft.fftshift(hmap)
		hmap = np.concatenate([hmap, hmap[:1, :]], axis=0)
		hmap = np.concatenate([hmap, hmap[:, :1]], axis=1)
		if smooth > 0:
			sigma = spectrum.shape[0] / image_size * smooth
			hmap = scipy.ndimage.gaussian_filter(hmap, sigma=sigma, mode='nearest')
		
		if power == False:
			hmap = 10 ** (hmap / 10)
		hmap_dict[i_key] = hmap
		image_size_dict[i_key] = image_size
		
	return hmap_dict, image_size_dict

def construct_std_heatmap(mean_npz_file, npz_file, smooth, mean_key='spectrum', std_key='spectrum'):
	mean_npz_data = np.load(mean_npz_file)
	npz_data = np.load(npz_file)
	std_spectrum = np.sqrt(npz_data[std_key])
	mean_spectrum = mean_npz_data[mean_key]
	image_size = npz_data['image_size']
	up_hmap = np.log10(mean_spectrum + std_spectrum) * 10
	low_hmap = np.log10((mean_spectrum - std_spectrum).clip(min=1e-50)) * 10
	# hmap = np.log10(std_spectrum / (mean_spectrum) ** 2 + 1) * 100
	# hmap = np.log10(spectrum) * 10 # dB
	up_hmap = np.fft.fftshift(up_hmap)
	up_hmap = np.concatenate([up_hmap, up_hmap[:1, :]], axis=0)
	up_hmap = np.concatenate([up_hmap, up_hmap[:, :1]], axis=1)
	if smooth > 0:
		sigma = std_spectrum.shape[0] / image_size * smooth
		up_hmap = scipy.ndimage.gaussian_filter(up_hmap, sigma=sigma, mode='nearest')

	low_hmap = np.fft.fftshift(low_hmap)
	low_hmap = np.concatenate([low_hmap, low_hmap[:1, :]], axis=0)
	low_hmap = np.concatenate([low_hmap, low_hmap[:, :1]], axis=1)
	if smooth > 0:
		sigma = std_spectrum.shape[0] / image_size * smooth
		low_hmap = scipy.ndimage.gaussian_filter(low_hmap, sigma=sigma, mode='nearest')

	return up_hmap, low_hmap, image_size

#----------------------------------------------------------------------------

@click.group()
def main():
	"""Compare average power spectra between real and generated images,
	or between multiple generators.

	Example:

	\b
	# Calculate dataset mean and std, needed in subsequent steps.
	python discriminator_test.py test --source=~/datasets/ffhq-1024x1024.zip

	"""

#----------------------------------------------------------------------------

class NoiseGenerator(torch.nn.Module):
	def __init__(self):
		super().__init__()
	
	def forward(self):
		print('test')

@main.command()
@click.option('--source', help='dataset zip, or directory', metavar='[ZIP|DIR]', required=True)
@click.option('--mean', help='NPZ', metavar='[NPZ]', required=True)
@click.option('--std', help='NPZ', metavar='[NPZ]', required=True)
@click.option('--network', help='Network to test', metavar='PKL', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--nettype', help='Network type', type=click.Choice(["baseline", "ours"]), default="baseline", show_default=True)
def prune_test(source, mean, std, network, dest, num, seed, nettype, device=torch.device('cuda')):
	"""Calculate per frequency Noise/Gradient map similarity in .npz file."""
	torch.multiprocessing.set_start_method('spawn')
	_, image_size, real_iter = stream_source_images(source=source, num=None, seed=seed, device=device, batch_size=16)
	_, image_size, fake_iter = stream_source_images(source=network, num=15000, seed=seed, device=device, batch_size=16)

	with dnnlib.util.open_url(network) as f:
		network_dict = legacy.load_network_pkl(f)
		D = network_dict['D'].to(device)
		# G = network_dict['G_ema'].to(device)
	
	target_mask = dict()
	param_dict = dict(D.named_parameters())

	if nettype == 'baseline':
		for res in D.block_resolutions:
			conv0_weight = param_dict[f'b{res}.conv0.weight']
			target_mask[f'b{res}.conv0.weight'] = torch.zeros(*conv0_weight.shape[:2])
			conv1_weight = param_dict[f'b{res}.conv1.weight']
			target_mask[f'b{res}.conv1.weight'] = torch.zeros(*conv1_weight.shape[:2])

	elif nettype == "ours":
		for res in D.block_resolutions:
			conv0_weight = getattr(D, f'b{res}').conv0_weight()
			target_mask[f'b{res}.conv0.weight'] = torch.zeros(*conv0_weight.shape[:2])
			conv1_weight = getattr(D, f'b{res}').conv1_weight()
			target_mask[f'b{res}.conv1.weight'] = torch.zeros(*conv1_weight.shape[:2])

	for real_image in tqdm.tqdm(real_iter, total=num):
		real_image = (real_image.to(torch.float32) - 127.5) / 127.5
		x = D.b128.fromrgb(real_image)


		for res in D.block_resolutions:
			block = getattr(D, f'b{res}')
			if nettype == 'baseline':
				y = block.skip(x, gain=np.sqrt(0.5))
				test_x = block.conv0(x)
				prune.ln_structured(block.conv0, 'weight', 0.1,  n=2, dim=1)
				block.conv0.weight.weight_mask = (block.conv0.weight.norm(dim=[2,3], keepdim=True) > 1).float().repeat(1,1,3,3)
				prune_x = block.conv0(x)


			elif nettype == 'ours':
				pass

		
	# Save result.
	if os.path.dirname(dest):
		os.makedirs(os.path.dirname(dest), exist_ok=True)
	np.savez(dest, spectrum=spectrum.cpu().numpy(), image_size=image_size)

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='stat jsonl dir', metavar='DIR', required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
def stats(source, dest, num, seed, device=torch.device('cuda')):
	"""Calculate dataset mean and standard deviation needed by 'calc'."""
	torch.multiprocessing.set_start_method('spawn')
	num_images, _image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device, batch_size=1)

	# Accumulate moments.
	moments = torch.zeros([3], dtype=torch.float64, device=device)
	for image in tqdm.tqdm(image_iter, total=num_images):
		image = image.to(torch.float64)
		save_image(image / 255, 'output/act_mask/image.png')
		fft_image = torch.fft.fftshift(torch.fft.fft2(image/255))
		fft_image_abs = (fft_image.abs() + 1).log10()
		fft_image_abs /= fft_image_abs.max()
		for i in range(255):
			act_mask = ((image - i) > 0).float()
			fft_act_mask = torch.fft.fftshift(torch.fft.fft2(act_mask))
			fft_act_mask = (fft_act_mask.abs() + 1).log10()
			fft_act_mask /= fft_act_mask.max()
			save_image(torch.cat([fft_image_abs, fft_act_mask, act_mask * image / 255], dim=-1), f'output/act_mask/{i}.png')
			
		moments += torch.stack([torch.ones_like(image).sum(), image.sum(), image.square().sum()])
	moments = moments / moments[0]

	# Compute mean and standard deviation.
	mean = moments[1]
	std = (moments[2] - moments[1].square()).sqrt()
	print(f'--mean={mean:g} --std={std:g}')

	filename, ext = os.path.splitext(source)
	if ext == '.zip':
		filename = os.path.basename(filename)
		result_dict = {"name": filename, "mean": mean.item(), "std": std.item()}
		jsonl_line = json.dumps(result_dict)
		print(jsonl_line)
		with open(os.path.join(dest, f'stats.jsonl'), 'at') as f:
			f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='path to npz', metavar='DIR', required=True)
@click.option('--interp', help='inerpolation rate', metavar='INT', default=1, show_default=True, type=click.IntRange(min=1))
def calc_phase(source, dest, interp, device=torch.device('cuda')):
	"""Calculating distribution of phase per frequency."""
	torch.multiprocessing.set_start_method('spawn')
	# num_hist = 12
	num_hist = 2
	with dnnlib.util.open_url(source) as f:
		D = legacy.load_network_pkl(f)['D'].to(device)
	

	res = max(D.block_resolutions)
	# whole_spectrum = torch.zeros([num_hist ** 3, res * interp, res * interp], dtype=torch.float16, device=device)
	# whole_spectrum = torch.zeros([num_hist, res * interp, res * interp], dtype=torch.float16, device=device)
	# whole_count = 0
	spectrum_dict = {}
	center = res * interp // 2
	for res in D.block_resolutions:
		block = getattr(D, f'b{res}')

		# block_spectrum = torch.zeros([num_hist ** 3, res * interp, res * interp], dtype=torch.float16, device=device)
		block_spectrum = torch.zeros([res * interp, res * interp, 2], dtype=torch.float16, device=device)
		num_freqs = torch.ones([res * interp, res * interp], dtype=torch.float16, device=device) * 1e-6

		conv0_freq = block.conv0_weight.get_freqs()
		conv0_freq = (((conv0_freq + res / 2).remainder(res) - res / 2) * interp).floor().long().flatten(0,1) + res * interp // 2
		# conv0_phase = (block.conv0_weight.phases * 2 * np.pi + 2 * np.pi).remainder(2 * np.pi) * (num_hist ** 3) / (2 * np.pi)
		conv0_phase = torch.cat([(block.conv0_weight.phases * 2 * np.pi).sin().unsqueeze(-1), (block.conv0_weight.phases * 2 * np.pi).cos().unsqueeze(-1)], dim=-1)
		# conv0_phase = conv0_phase.long()
		# block_spectrum[conv0_phase.flatten(), conv0_freq[:,0], conv0_freq[:,1]] += 1
		block_spectrum[conv0_freq[:,0], conv0_freq[:,1], :] += conv0_phase.flatten(0,1)
		for i in range(len(conv0_freq)):
			num_freqs[conv0_freq[i,0], conv0_freq[i,1]] += 1
		block_mean = block_spectrum / num_freqs.unsqueeze(-1)

		conv0_std = (conv0_phase.flatten(0, 1) - block_mean[conv0_freq[:,0], conv0_freq[:,1], :]).square()
		block_std = torch.zeros_like(block_mean)
		for i in range(len(conv0_freq)):
			block_std[conv0_freq[i,0], conv0_freq[i,1],:] = conv0_std[i]
		block_std = block_std / num_freqs.unsqueeze(-1)
		block_std[num_freqs == 1e-6, :] = 0
		spectrum_dict[f"conv0_{res}_std"] = block_std.sum(dim=-1).cpu()

		block_spectrum = torch.zeros([res * interp, res * interp, 2], dtype=torch.float16, device=device)
		num_freqs = torch.ones([res * interp, res * interp], dtype=torch.float16, device=device) * 1e-6

		conv1_freq = block.conv1_weight.get_freqs()
		conv1_freq = (((conv1_freq + res / 2).remainder(res) - res / 2) * interp).floor().long().flatten(0,1) + res * interp // 2
		# conv1_phase = (block.conv1_weight.phases * 2 * np.pi + 2 * np.pi).remainder(2 * np.pi) * (num_hist ** 3) / (2 * np.pi)
		conv1_phase = torch.cat([(block.conv1_weight.phases * 2 * np.pi).sin().unsqueeze(-1), (block.conv1_weight.phases * 2 * np.pi).cos().unsqueeze(-1)], dim=-1)
		# conv1_phase = conv1_phase.long()
		# block_spectrum[conv1_phase.flatten(), conv1_freq[:,0], conv1_freq[:,1]] += 1
		block_spectrum[conv1_freq[:,0], conv1_freq[:,1], :] += conv1_phase.flatten(0,1)
		for i in range(len(conv1_freq)):
			num_freqs[conv1_freq[i,0], conv1_freq[i,1]] += 1
		block_mean = block_spectrum / num_freqs.unsqueeze(-1)

		conv1_std = (conv1_phase.flatten(0, 1) - block_mean[conv1_freq[:,0], conv1_freq[:,1], :]).square()
		block_std = torch.zeros_like(block_mean)
		for i in range(len(conv1_freq)):
			block_std[conv1_freq[i,0], conv1_freq[i,1],:] = conv1_std[i]
		block_std = block_std / num_freqs.unsqueeze(-1)
		block_std[num_freqs == 1e-6, :] = 0
		spectrum_dict[f"conv1_{res}_std"] = block_std.sum(dim=-1).cpu()

	# whole_spectrum /= whole_count
	# whole_spectrum = whole_spectrum.clip(min=1e-6)
	# whole_entropy = (-whole_spectrum * whole_spectrum.log()).sum(dim=0)

	# spectrum_dict['whole'] = whole_entropy

	dir_name = os.path.join(os.path.dirname(source), dest)
	pkl_num = re.findall(r'\d+', source)[-1]
	if dir_name:
		os.makedirs(dir_name, exist_ok=True)
	
	np.savez(os.path.join(dir_name, f'{pkl_num}.npz'), **spectrum_dict)

@torch.no_grad()
@main.command()
@click.option('--source', help='Network pkl', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--interp', help='inerpolation rate', metavar='INT', default=1, show_default=True, type=click.IntRange(min=1))
def calc_mag_inception(source, interp, device=torch.device('cuda')):
	"""Calculating distribution of phase per frequency."""
	torch.multiprocessing.set_start_method('spawn')
	dest = "Network_magnitude"
	dir_name = os.path.join("output/metric/vgg", dest)
	if dir_name:
		os.makedirs(dir_name, exist_ok=True)

	conv_weight = lambda x : x.weight 

	def fft_weight(block, weight_key, res, device):
		weight = weight_key(block)[:10]
		ks = weight.shape[-1]
		pad_size = (res - ks) // 2 + 1
		weight = weight * np.sqrt(weight.shape[1]) * ks 
		weight = torch.nn.functional.pad(weight, [pad_size] * 4)
		weight = torch.fft.fftshift(torch.fft.fft2(weight.to(device)))
		w_freq_set = torch.fft.fftfreq(weight.shape[-2]).view(1,-1, 1).repeat(weight.shape[-2], 1, 1)
		h_freq_set = torch.fft.fftfreq(weight.shape[-2]).view(-1,1, 1).repeat(1, weight.shape[-1], 1)
		freq_set = torch.cat([w_freq_set, h_freq_set], dim=-1)
		return weight, freq_set
	
	def display_image(fft_mag, res_key, conv_key, pkl_num):
		mean_fp = os.path.join(dir_name,f'{conv_key}_mean.png')
		# std_fp = os.path.join(dir_name,f'{pkl_num}_{res_key}_{conv_key}_std.png')

		# fft_mag = fft_mag.abs().log10().flatten(0,1)
		# fft_mag = (fft_mag - fft_mag.min())
		# fft_mag = fft_mag / fft_mag.max()
		# save_image(fft_mag.mean(dim=0),
		# 		os.path.join(dir_name, )
		# save_image(fft_mag.std(dim=0),
		# 		os.path.join(dir_name, f'{pkl_num}_{res_key}_{conv_key}_std.png'))
		# fft_mag = fft_mag.to(torch.device("cpu"))
		plt.clf()
		mean_fft = fft_mag.abs().flatten(0,1).mean(dim=0)[res_key // 2 + 1,:].to("cpu")
		std_fft = fft_mag.abs().flatten(0,1).std(dim=0)[res_key // 2 + 1,:].to("cpu")
		max_fft = (mean_fft + std_fft).log10().max().item()
		plt.plot(torch.arange(len(mean_fft)), (mean_fft + std_fft).log10(), c='r')
		plt.plot(torch.arange(len(mean_fft)), (mean_fft - std_fft).clip(min=1e-10).log10(), c='b')
		plt.text(0,0.4, f'{res_key}_{conv_key}_{max_fft}')
		plt.savefig(mean_fp)

	detector = get_feature_detector(url=source, device=device, num_gpus=1, rank=0, verbose=True)
	x = torch.randn([1,3,64,64], device=device)
	for i, layer in enumerate(detector.layers):
		img_size = x.shape[-1]
		module_dict = dict(detector.layers[layer].named_modules())
		for k, v in module_dict.items():
			if hasattr(v, '_orig_class_name') and (v._orig_class_name == 'Conv2dLayer') and (v.weight.shape[-1] >= 3) and (v.weight.shape[-1] == v.weight.shape[-2]):
				fft_mag, fft_freq_set = fft_weight(v, conv_weight, img_size, device)
				display_image(fft_mag, img_size, f'layer{i}_{k}_{img_size}', "00000")
			else:
				continue

#----------------------------------------------------------------------------

@torch.no_grad()
@main.command()
@click.option('--source', help='Network pkl', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--interp', help='inerpolation rate', metavar='INT', default=1, show_default=True, type=click.IntRange(min=1))
def calc_mag(source, interp, device=torch.device('cuda')):
	"""Calculating distribution of phase per frequency."""
	torch.multiprocessing.set_start_method('spawn')
	dest = "Network_magnitude"
	pkl_num = re.findall(r'\d+', source)[-1]
	dir_name = os.path.join(os.path.dirname(source), dest)
	if dir_name:
		os.makedirs(dir_name, exist_ok=True)

	baseline_network = ("baseline" in source) or ("pretrained" in source)
	# baseline_network = True
	conv0_weight = lambda x, k : x.conv0.weight if baseline_network else x.conv0_weight(torch.device("cpu"),k) / x.conv0_weight.gain(k)
	conv1_weight = lambda x, k : x.conv1.weight if baseline_network else x.conv1_weight(torch.device("cpu"),k) / x.conv1_weight.gain(k)

	def fft_weight(block, weight_key, res, ks, device):
		pad_size = res // 2
		weight = weight_key(block, ks)[:100]
		weight = torch.nn.functional.pad(weight, [pad_size] * 4)
		weight = torch.fft.fftshift(torch.fft.fft2(weight.to(device)))
		w_freq_set = torch.fft.fftfreq(weight.shape[-2]).view(1,-1, 1).repeat(weight.shape[-2], 1, 1)
		h_freq_set = torch.fft.fftfreq(weight.shape[-2]).view(-1,1, 1).repeat(1, weight.shape[-1], 1)
		freq_set = torch.cat([w_freq_set, h_freq_set], dim=-1)
		return weight, freq_set
	
	def display_image(fft_mag, res_key, conv_key, pkl_num):
		mean_fp = os.path.join(dir_name,f'{res_key}_{conv_key}',f'{pkl_num}.png')
		os.makedirs(os.path.dirname(mean_fp), exist_ok=True)
		# std_fp = os.path.join(dir_name,f'{pkl_num}_{res_key}_{conv_key}_std.png')

		# fft_mag = fft_mag.abs().log10().flatten(0,1)
		# fft_mag = (fft_mag - fft_mag.min())
		# fft_mag = fft_mag / fft_mag.max()
		# save_image(fft_mag.mean(dim=0),
		# 		os.path.join(dir_name, )
		# save_image(fft_mag.std(dim=0),
		# 		os.path.join(dir_name, f'{pkl_num}_{res_key}_{conv_key}_std.png'))
		# fft_mag = fft_mag.to(torch.device("cpu"))
		plt.clf()
		plt.subplot(1,2,1)
		# mean_fft = fft_mag.abs().flatten(0,1).mean(dim=0)[res_key // 2 + 1,:].to("cpu")
		std_fft = fft_mag.abs().flatten(0,1).std(dim=0).to("cpu")
		# plt.plot(torch.arange(len(mean_fft)), (mean_fft + std_fft).log10(), c='r')
		# plt.plot(torch.arange(len(mean_fft)), (mean_fft - std_fft).clip(min=1e-10).log10(), c='b')
		plt.plot(torch.arange(len(std_fft) // 2), std_fft[res_key // 2 + 1,len(std_fft)//2+1:], c='r')
		plt.text(0,0.4, f'{res_key}_{conv_key}')
		plt.subplot(1,2,2)
		plt.plot(torch.arange(len(std_fft)// 2), std_fft[torch.arange(len(std_fft)//2) + len(std_fft)//2, torch.arange(len(std_fft)//2) + len(std_fft)//2], c='r')
		plt.savefig(mean_fp)

	with dnnlib.util.open_url(source) as f:
		D = legacy.load_network_pkl(f)['D']

	res = max(D.block_resolutions)
	for res in D.block_resolutions:
		block = getattr(D, f'b{res}')

		fft_mag, fft_freq_set = fft_weight(block, conv0_weight, res, 3, device)
		display_image(fft_mag, res, 'conv0', pkl_num)
		fft_mag = fft_mag.to("cpu")
		fft_mag, fft_freq_set = fft_weight(block, conv1_weight, res, 3, device)
		display_image(fft_mag, res, 'conv1', pkl_num)
		fft_mag = fft_mag.to("cpu")

		del block


	# np.savez(os.path.join(dir_name, f'{pkl_num}.npz'), **spectrum_dict)

#----------------------------------------------------------------------------

@main.command()
@click.argument('npz-file', nargs=1)
@click.option('--save', help='Save the plot and exit', metavar='[PNG|PDF|...]')
@click.option('--dpi', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=100, show_default=True)
@click.option('--smooth', help='Amount of smoothing', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
def azimuthal_entropy(npz_file, save, smooth, dpi):
	"""Visualize 2D heatmap based on the given .npz file."""
	hmap_dict, image_size_dict = construct_heatmap(npz_file=npz_file, smooth=smooth, key='spectrum_entropy', power=False)
	pkl_num = re.findall(r'\d+', npz_file)[-1]

	# Setup plot.
	for k in hmap_dict.keys():
		hmap = hmap_dict[k]
		image_size = image_size_dict[k]
		plt.clf()
		plt.figure(figsize=[6, 4.8], dpi=dpi, tight_layout=True)
		freqs = np.linspace(0, 1, num=int(image_size * np.sqrt(2) // 2), endpoint=True) * image_size * np.sqrt(2) // 2
		# ticks = np.linspace(freqs[0], freqs[-1], num=5, endpoint=True)

		map_idx = torch.arange(end=hmap.shape[-1])
		map_idx = torch.cat([map_idx.view(-1, 1, 1).repeat(1, hmap.shape[-1], 1), map_idx.view(1, -1, 1).repeat(hmap.shape[-1],1 , 1)], dim=-1) - hmap.shape[-1] // 2

		freq_norm_id = map_idx.square().sum(dim=-1).sqrt().floor()
		
		per_dist_entropy = torch.zeros(len(freqs))
		for f_id, freq in enumerate(freqs):
			round_dist = hmap[freq_norm_id == f_id]
			if len(round_dist):
				per_dist_entropy[f_id] = float(round_dist.mean())

		# Draw plot.
		plt.xlim(freqs[0], freqs[-1])
		plt.ylim(per_dist_entropy.min(), per_dist_entropy.max())
		plt.xticks(freqs[::20])
		plt.yticks(np.linspace(per_dist_entropy.min(), per_dist_entropy.max(), num=10, endpoint=True))
		plt.plot(freqs, per_dist_entropy)

		# Display or save.
		if save is None:
			plt.show()
		else:
			if os.path.dirname(npz_file):
				os.makedirs(os.path.join(os.path.dirname(npz_file), save), exist_ok=True)
			save_path = os.path.join(os.path.dirname(npz_file), save, f'{pkl_num}_{k}.png')
			plt.savefig(save_path)

#----------------------------------------------------------------------------

if __name__ == "__main__":
	main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------