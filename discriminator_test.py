import os
import numpy as np
import torch
import torch.fft
import scipy.ndimage
import matplotlib.pyplot as plt
import click
import tqdm
import dnnlib
import json

import legacy
from training import dataset

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

def construct_heatmap(npz_file, smooth, key='spectrum'):
	npz_data = np.load(npz_file)
	spectrum = npz_data[key]
	image_size = npz_data['image_size']
	hmap = np.log10(spectrum) * 10 # dB
	hmap = np.fft.fftshift(hmap)
	hmap = np.concatenate([hmap, hmap[:1, :]], axis=0)
	hmap = np.concatenate([hmap, hmap[:, :1]], axis=1)
	if smooth > 0:
		sigma = spectrum.shape[0] / image_size * smooth
		hmap = scipy.ndimage.gaussian_filter(hmap, sigma=sigma, mode='nearest')
	return hmap, image_size

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

@main.command()
@click.option('--source', help='dataset zip, or directory', metavar='[ZIP|DIR]', required=True)
@click.option('--mean', help='NPZ', metavar='[NPZ]', required=True)
@click.option('--std', help='NPZ', metavar='[NPZ]', required=True)
@click.option('--network', help='Network to test', metavar='PKL', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
def noise_test(source, mean, std, network, dest, num, seed, device=torch.device('cuda')):
	"""Calculate per frequency Noise/Gradient map similarity in .npz file."""
	torch.multiprocessing.set_start_method('spawn')
	_, image_size, image_iter = stream_source_images(source=source, num=None, seed=seed, device=device, batch_size=64)

	# Test discriminator with per frequency noise.
	freq = torch.fft.fftshift(torch.fft.fftfreq(n=image_size, device=device)) * image_size
	freq_set = torch.cat([freq.unsqueeze(-1).repeat(1, image_size).unsqueeze(-1), freq.unsqueeze(0).repeat(image_size, 1).unsqueeze(-1)], dim=-1)[:image_size//2, ...].flatten(0,1)
	# Construct sampling grid.
	theta = torch.eye(2, 3, device=device)
	grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, image_size, image_size], align_corners=False).to(device=device)
	
	for image in tqdm.tqdm(image_iter, total=num):
		freq_id = torch.randint(high=len(freq_set), size=[64])
		cur_freq = freq_set[freq_id]
		cur_phase = torch.rand([len(freq_id), 1, 1], device=device)
		cur_noise = ((torch.einsum('bhwr,fr->fhw',grids, cur_freq) + cur_phase) * np.pi * 2).sin()
		image = (image.to(torch.float64) - 127.5) / 127.5
		

		image = torch.nn.functional.pad(image * window, [0, padding, 0, padding])
		spectrum += torch.fft.fftn(image, dim=[2,3]).abs().square().mean(dim=[0, 1]).sqrt()
	spectrum /= num_images

	# Save result.
	if os.path.dirname(dest):
		os.makedirs(os.path.dirname(dest), exist_ok=True)
	np.savez(dest, spectrum=spectrum.cpu().numpy(), image_size=image_size)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
