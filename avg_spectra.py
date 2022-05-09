# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Compare average power spectra between real and generated images,
or between multiple generators."""

import os
from random import random
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
from metrics.metric_utils import FeatureStats
from training import dataset

def azimuthal_average(image, center=None):
    # modified to tensor inputs from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """
    Calculate the azimuthally averaged radial profile.
    Requires low frequencies to be at the center of the image.
    Args:
        image: Batch of 2D images, NxHxW
        center: The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    Returns:
    Azimuthal average over the image around the center
    """
    # Check input shapes
    assert center is None or (len(center) == 2), f'Center has to be None or len(center)=2 ' \
                                                 f'(but it is len(center)={len(center)}.'
    # Calculate the indices from the image
    H, W = image.shape[-2:]
    h, w = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    if center is None:
        center = torch.tensor([(w.max() - w.min()) / 2.0, (h.max() - h.min()) / 2.0])

    # Compute radius for each pixel wrt center
    r = torch.stack([w - center[0], h - center[1]]).norm(2, 0)

    # Get sorted radii
    r_sorted, ind = r.flatten().sort()
    i_sorted = image.flatten(-2, -1)[..., ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.long()  # attribute to the smaller integer

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented, computes bin change between subsequent radii
    rind = torch.where(deltar)[0]  # location of changed radius

    # compute number of elements in each bin
    nind = rind + 1  # number of elements = idx + 1
    nind = torch.cat([torch.tensor([0]), nind, torch.tensor([H * W])])  # add borders
    nr = nind[1:] - nind[:-1]  # number of radius bin, i.e. counter for bins belonging to each radius

    # Cumulative sum to figure out sums for each radius bin
    if H % 2 == 0:
        raise NotImplementedError('Not sure if implementation correct, please check')
        rind = torch.cat([torch.tensor([0]), rind, torch.tensor([H * W - 1])])  # add borders
    else:
        rind = torch.cat([rind, torch.tensor([H * W - 1])])  # add borders
    csim = i_sorted.cumsum(-1, dtype=torch.float64)  # integrate over all values with smaller radius
    tbin = csim[..., rind[1:]] - csim[..., rind[:-1]]
    # add mean
    tbin = torch.cat([csim[:, 0:1], tbin], 1)

    radial_prof = tbin / nr.to(tbin.device)  # normalize by counted bins

    return radial_prof

#----------------------------------------------------------------------------
# Setup an iterator for streaming images, in uint8 NCHW format, based on the
# respective command line options.

def stream_source_images(source, num, seed, device, data_loader_kwargs=None): # => num_images, image_size, image_iter
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
        data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=1, **data_loader_kwargs)
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
    # up_hmap = mean_spectrum + std_spectrum
    # low_hmap = mean_spectrum - std_spectrum
    
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
    python avg_spectra.py stats --source=~/datasets/ffhq-1024x1024.zip

    \b
    # Calculate average spectrum for the training data.
    python avg_spectra.py calc --source=~/datasets/ffhq-1024x1024.zip \\
        --dest=tmp/training-data.npz --mean=112.684 --std=69.509
        115.673, 63.940

    \b
    # Calculate average spectrum for a pre-trained generator.
    python avg_spectra.py calc \\
        --source=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl \\
        --dest=tmp/stylegan3-r.npz --mean=112.684 --std=69.509 --num=70000

    \b
    # Calculate standard deviation of spectrum for a pre-trained generator.
    python avg_spectra.py calc_std \\
        --source=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl \\
        --dest=tmp/stylegan3-r.npz --npz_mean=tmp/stylegan3-r.npz --mean=112.684 --std=69.509 --num=70000

    \b
    # Display results.
    python avg_spectra.py heatmap tmp/training-data.npz
    python avg_spectra.py heatmap tmp/stylegan3-r.npz
    python avg_spectra.py slices tmp/training-data.npz tmp/stylegan3-r.npz

    \b
    # Save as PNG.
    python avg_spectra.py heatmap tmp/training-data.npz --save=tmp/training-data.png --dpi=300
    python avg_spectra.py heatmap tmp/stylegan3-r.npz --save=tmp/stylegan3-r.png --dpi=300
    python avg_spectra.py slices tmp/training-data.npz tmp/stylegan3-r.npz --save=tmp/slices.png --dpi=300
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='stat jsonl dir', metavar='DIR', required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
def stats(source, dest, num, seed, device=torch.device('cuda')):
    """Calculate dataset mean and standard deviation needed by 'calc'."""
    torch.multiprocessing.set_start_method('spawn')
    num_images, _image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device)

    # Accumulate moments.
    moments = torch.zeros([3], dtype=torch.float64, device=device)
    for image in tqdm.tqdm(image_iter, total=num_images):
        image = image.to(torch.float64)
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
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--network', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--mean', help='Dataset mean for whitening', metavar='FLOAT', type=float, required=True)
@click.option('--std', help='Dataset standard deviation for whitening', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--beta', help='Shape parameter for the Kaiser window', metavar='FLOAT', type=click.FloatRange(min=0), default=8, show_default=True)
@click.option('--interp', help='Frequency-domain interpolation factor', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
def calc_mean_cov(source, network, dest, mean, std, num, seed, beta, interp, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    torch.multiprocessing.set_start_method('spawn')
    num_dataset, image_size, image_iter = stream_source_images(source=source, seed=seed, device=device, num=num)
    num_gen, image_size, generator_iter = stream_source_images(source=network, seed=seed, device=device, num=num)
    spectrum_size = image_size * interp
    padding = (spectrum_size - image_size) + 1
    dataset_stat = FeatureStats(capture_mean_cov=True)
    generator_stat = FeatureStats(capture_mean_cov=True)

    # Setup window function.
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0).unsqueeze(1)

    # Accumulate power spectrum.
    for image in tqdm.tqdm(image_iter, total=num_dataset):
        image = (image.to(torch.float64) - mean) / std
        image = torch.nn.functional.pad(image * window, [0, padding, 0, padding])
        fft_image = torch.fft.fftshift(torch.fft.fftn(image, dim=[2,3]).abs().square().mean(dim=[0,1]).sqrt())
        azim_image = azimuthal_average(fft_image.unsqueeze(0))
        azim_image = azim_image / azim_image[0,0]
        # aff_len = int(azim_image[0].shape[0] * 0.75)
        aff_len = 0
        dataset_stat.append_torch(azim_image[:,aff_len:])

    mu_real, sigma_real = dataset_stat.get_mean_cov()
    mu_real = torch.tensor(mu_real).cuda()
    
    sd = 0
    for image in tqdm.tqdm(generator_iter, total=num_gen):
        image = (image.to(torch.float64) - mean) / std
        image = torch.nn.functional.pad(image * window, [0, padding, 0, padding])
        fft_image = torch.fft.fftshift(torch.fft.fftn(image, dim=[2,3]).abs().square().mean(dim=[0,1]).sqrt())
        azim_image = azimuthal_average(fft_image.unsqueeze(0))
        azim_image = azim_image / azim_image[0,0]
        # aff_len = int(azim_image[0].shape[0] * 0.75)
        aff_len = 0
        sd_i = -(mu_real * torch.log(azim_image + 1e-8) + (1 - mu_real) * (torch.log(1 - azim_image + 1e-8))).mean()
        assert sd_i.isfinite().item()
        sd += sd_i

        # generator_stat.append_torch(azim_image[:,aff_len:])
    sd /= num_gen
    print(sd)
    
    mu_gen, sigma_gen = dataset_stat.get_mean_cov()
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    print(fid)

    # Save result.
    if os.path.dirname(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--network', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--mean', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--beta', help='Shape parameter for the Kaiser window', metavar='FLOAT', type=click.FloatRange(min=0), default=8, show_default=True)
@click.option('--interp', help='Frequency-domain interpolation factor', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
def calc_feat_std(source, network, mean, dest, num, beta, interp, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    torch.multiprocessing.set_start_method('spawn')
    mean_dict = np.load(mean)
    key_list = [k.split('.')[0] for k in mean_dict._files]
    spectrum_mean_dict = {k: torch.tensor(mean_dict[k]).cuda() for k in key_list}

    seed = int(random() * 10000)
    num_images, image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device)
    mean = 127.5
    std = 127.5

    with dnnlib.util.open_url(network) as f:
        D = legacy.load_network_pkl(f)['D'].to(device)

    spectrum_size = image_size * interp
    padding = (spectrum_size - image_size)


    num_spectrum = 1 + len(D.block_resolutions)

    # Setup window function.
    window_dict = {}
    spectrum_dict = {}
    
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0).unsqueeze(1)
    window_dict["image"] = window
    spectrum_dict["image"] = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)

    for res in D.block_resolutions:
        window = torch.kaiser_window(res//2, periodic=False, beta=beta, device=device)
        window *= window.square().sum().rsqrt()
        window = window.ger(window).unsqueeze(0).unsqueeze(1)
        window_dict[res] = window
        spectrum_dict[f'{res}'] = torch.zeros([res//2, res//2], dtype=torch.float64, device=device)

    # Accumulate power spectrum.
    for image in tqdm.tqdm(image_iter, total=num_images):
        image = (image - mean) / std
        image64 = image.to(torch.float64)
        image64 = torch.nn.functional.pad(image64 * window_dict["image"], [0, padding, 0, padding])
        mag_image = torch.fft.fftn(image64, dim=[2,3]).abs()
        spectrum_dict["image"] += (mag_image - spectrum_mean_dict["image"]).square().mean(dim=[0,1])
        x = None
        for i, res in enumerate(D.block_resolutions):
            block = getattr(D, f'b{res}')
            x, image = block(x, image)
            x64 = x.to(torch.float64)
            x64 = torch.nn.functional.pad(x64 * window_dict[res], [0, padding, 0, padding])
            mag_x = torch.fft.fftn(x64, dim=[2,3]).abs()
            spectrum_dict[f'{res}'] += (mag_x - spectrum_mean_dict[f'{res}']).square().mean(dim=[0,1])

    for k in list(spectrum_dict.keys()):
        spectrum_dict[k] = spectrum_dict[k].cpu().numpy() / num_images

    # Save result.
    if os.path.dirname(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, image_size=image_size, **spectrum_dict)

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--network', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--beta', help='Shape parameter for the Kaiser window', metavar='FLOAT', type=click.FloatRange(min=0), default=8, show_default=True)
@click.option('--interp', help='Frequency-domain interpolation factor', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
def calc_feat_mean(source, network, dest, num, beta, interp, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    torch.multiprocessing.set_start_method('spawn')
    seed = int(random() * 10000)
    num_images, image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device)
    mean = 127.5
    std = 127.5

    with dnnlib.util.open_url(network) as f:
        D = legacy.load_network_pkl(f)['D'].to(device)

    spectrum_size = image_size * interp
    padding = (spectrum_size - image_size)


    num_spectrum = 1 + len(D.block_resolutions)

    # Setup window function.
    window_dict = {}
    spectrum_dict = {}
    
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0).unsqueeze(1)
    window_dict["image"] = window
    spectrum_dict["image"] = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)

    for res in D.block_resolutions:
        window = torch.kaiser_window(res//2, periodic=False, beta=beta, device=device)
        window *= window.square().sum().rsqrt()
        window = window.ger(window).unsqueeze(0).unsqueeze(1)
        window_dict[res] = window
        spectrum_dict[f'{res}'] = torch.zeros([res//2, res//2], dtype=torch.float64, device=device)

    # Accumulate power spectrum.
    for image in tqdm.tqdm(image_iter, total=num_images):
        image = (image - mean) / std
        image64 = image.to(torch.float64)
        image64 = torch.nn.functional.pad(image64 * window_dict["image"], [0, padding, 0, padding])
        spectrum_dict["image"] += torch.fft.fftn(image64, dim=[2,3]).abs().mean(dim=[0, 1])
        x = None
        for i, res in enumerate(D.block_resolutions):
            block = getattr(D, f'b{res}')
            x, image = block(x, image)
            x64 = x.to(torch.float64)
            x64 = torch.nn.functional.pad(x64 * window_dict[res], [0, padding, 0, padding])
            spectrum_dict[f'{res}'] += torch.fft.fftn(x64, dim=[2,3]).abs().mean(dim=[0, 1])


    for k in list(spectrum_dict.keys()):
        spectrum_dict[k] = spectrum_dict[k].cpu().numpy() / num_images

    # Save result.
    if os.path.dirname(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, image_size=image_size, **spectrum_dict)

#----------------------------------------------------------------------------


@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--mean', help='Dataset mean for whitening', metavar='FLOAT', type=float, required=True)
@click.option('--std', help='Dataset standard deviation for whitening', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--beta', help='Shape parameter for the Kaiser window', metavar='FLOAT', type=click.FloatRange(min=0), default=8, show_default=True)
@click.option('--interp', help='Frequency-domain interpolation factor', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
def calc(source, dest, mean, std, num, seed, beta, interp, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    torch.multiprocessing.set_start_method('spawn')
    num_images, image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device)
    spectrum_size = image_size * interp
    padding = (spectrum_size - image_size)

    # Setup window function.
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0).unsqueeze(1)

    # Accumulate power spectrum.
    spectrum = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)
    for image in tqdm.tqdm(image_iter, total=num_images):
        image = (image.to(torch.float64) - mean) / std
        image = torch.nn.functional.pad(image * window, [0, padding, 0, padding])
        spectrum += torch.fft.fftn(image, dim=[2,3]).abs().square().mean(dim=[0, 1]).sqrt()
    spectrum /= num_images

    # Save result.
    if os.path.dirname(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, spectrum=spectrum.cpu().numpy(), image_size=image_size)

#----------------------------------------------------------------------------

@main.command()
@click.option('--source', help='Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--npz_mean', help='NPZ file of mean', metavar='NPZ', required=True)
@click.option('--mean', help='Dataset mean for whitening', metavar='FLOAT', type=float, required=True)
@click.option('--std', help='Dataset standard deviation for whitening', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--beta', help='Shape parameter for the Kaiser window', metavar='FLOAT', type=click.FloatRange(min=0), default=8, show_default=True)
@click.option('--interp', help='Frequency-domain interpolation factor', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
def calc_std(source, dest, npz_mean, mean, std, num, seed, beta, interp, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    torch.multiprocessing.set_start_method('spawn')
    num_images, image_size, image_iter = stream_source_images(source=source, num=num, seed=seed, device=device)
    spectrum_size = image_size * interp
    padding = (spectrum_size - image_size)
    spectrum_mean = torch.Tensor(np.load(npz_mean)['spectrum']).cuda()

    # Setup window function.
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0).unsqueeze(1)

    # Accumulate power spectrum.
    spectrum = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)
    for image in tqdm.tqdm(image_iter, total=num_images):
        image = (image.to(torch.float64) - mean) / std
        image = torch.nn.functional.pad(image * window, [0, padding, 0, padding])
        spectrum += ((torch.fft.fftn(image, dim=[2,3])).abs().square().mean(dim=[0,1]).sqrt() -  spectrum_mean).square()
    spectrum /= num_images

    # Save result.
    if os.path.dirname(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, spectrum=spectrum.cpu().numpy(), image_size=image_size)

#----------------------------------------------------------------------------

@main.command()
@click.option('--prev', help='Previous Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--curr', help='Curr Network pkl, dataset zip, or directory', metavar='[PKL|ZIP|DIR]', required=True)
@click.option('--dest', help='Where to store the result', metavar='NPZ', required=True)
@click.option('--mean', help='Dataset mean for whitening', metavar='FLOAT', type=float, required=True)
@click.option('--std', help='Dataset standard deviation for whitening', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
@click.option('--num', help='Number of images to process  [default: all]', metavar='INT', type=click.IntRange(min=1))
@click.option('--seed', help='Random seed for selecting the images', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--beta', help='Shape parameter for the Kaiser window', metavar='FLOAT', type=click.FloatRange(min=0), default=8, show_default=True)
@click.option('--interp', help='Frequency-domain interpolation factor', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
def calc_diff(prev, curr, dest, mean, std, num, seed, beta, interp, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    torch.multiprocessing.set_start_method('spawn')
    num_images, image_size, prev_image_iter = stream_source_images(source=prev, num=num * 2, seed=seed, device=device)
    num_images, image_size, curr_image_iter = stream_source_images(source=curr, num=num * 2, seed=seed, device=device)
    spectrum_size = image_size * interp
    padding = spectrum_size - image_size

    # Setup window function.
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0).unsqueeze(1)

    # Accumulate power spectrum.
    spectrum_mean = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)
    counter = 0
    for prev_image, curr_image in tqdm.tqdm(zip(prev_image_iter, curr_image_iter), total=num_images//2):
        prev_image = (prev_image.to(torch.float64) - mean) / std
        prev_image = torch.nn.functional.pad(prev_image * window, [0, padding, 0, padding])

        curr_image = (curr_image.to(torch.float64) - mean) / std
        curr_image = torch.nn.functional.pad(curr_image * window, [0, padding, 0, padding])

        diff_image = curr_image - prev_image

        spectrum_mean += torch.fft.fftn(diff_image, dim=[2,3]).abs().square().mean(dim=[0,1])

        counter += 1
        if counter > (num_images // 2):
            break

    spectrum_mean /= (num_images // 2)

    spectrum_std = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)
    for prev_image, curr_image in tqdm.tqdm(zip(prev_image_iter, curr_image_iter), total=num_images//2):
        prev_image = (prev_image.to(torch.float64) - mean) / std
        prev_image = torch.nn.functional.pad(prev_image * window, [0, padding, 0, padding])

        curr_image = (curr_image.to(torch.float64) - mean) / std
        curr_image = torch.nn.functional.pad(curr_image * window, [0, padding, 0, padding])

        diff_image = curr_image - prev_image

        spectrum_std += (torch.fft.fftn(diff_image, dim=[2,3]).abs().square() - spectrum_mean).square().mean(dim=[0,1])

    spectrum_std /= (num_images//2)

    # Save result.
    if os.path.dirname(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, spectrum_mean=spectrum_mean.cpu().numpy(), spectrum_std=spectrum_std.cpu().numpy(), image_size=image_size)

#----------------------------------------------------------------------------

@main.command()
@click.argument('npz-file', nargs=1)
@click.option('--save', help='Save the plot and exit', metavar='[PNG|PDF|...]')
@click.option('--dpi', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=100, show_default=True)
@click.option('--smooth', help='Amount of smoothing', metavar='FLOAT', type=click.FloatRange(min=0), default=1.25, show_default=True)
def heatmap(npz_file, save, smooth, dpi):
    """Visualize 2D heatmap based on the given .npz file."""
    hmap, image_size = construct_heatmap(npz_file=npz_file, smooth=smooth)

    # Setup plot.
    plt.figure(figsize=[6, 4.8], dpi=dpi, tight_layout=True)
    freqs = np.linspace(-0.5, 0.5, num=hmap.shape[0], endpoint=True) * image_size
    ticks = np.linspace(freqs[0], freqs[-1], num=5, endpoint=True)
    levels = np.linspace(-40, 20, num=13, endpoint=True)

    # Draw heatmap.
    plt.xlim(ticks[0], ticks[-1])
    plt.ylim(ticks[0], ticks[-1])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.contourf(freqs, freqs, hmap, levels=levels, extend='both', cmap='Blues')
    plt.gca().set_aspect('equal')
    plt.colorbar(ticks=levels)
    plt.contour(freqs, freqs, hmap, levels=levels, extend='both', linestyles='solid', linewidths=1, colors='midnightblue', alpha=0.2)

    # Display or save.
    if save is None:
        plt.show()
    else:
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)

#----------------------------------------------------------------------------

@main.command()
@click.argument('npz-files', nargs=-1, required=True)
@click.option('--save', help='Save the plot and exit', metavar='[PNG|PDF|...]')
@click.option('--dpi', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=100, show_default=True)
@click.option('--smooth', help='Amount of smoothing', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--key', help='Figure resolution' , default="image", show_default=True)
@click.option('--ymax', help='Figure resolution', default=7.5, show_default=True)
@click.option('--ymin', help='Figure resolution', default=0, show_default=True)
def slices(npz_files, save, dpi, smooth, key, ymin, ymax):
    """Visualize 1D slices based on the given .npz files."""
    cases = [dnnlib.EasyDict(npz_file=npz_file) for npz_file in npz_files]
    ymax = 0
    ymin = 0
    for c in cases:
        c.hmap, c.image_size = construct_heatmap(npz_file=c.npz_file, smooth=smooth, key=key)
        if c.hmap.max() > ymax:
            ymax = c.hmap.max()
        if c.hmap.min() < ymin:
            ymin = c.hmap.min()
        c.label = os.path.splitext(os.path.basename(c.npz_file))[0]

    # Check consistency.
    image_size = cases[0].image_size
    hmap_size = cases[0].hmap.shape[0]
    if any(c.image_size != image_size or c.hmap.shape[0] != hmap_size for c in cases):
        raise click.ClickException('All .npz must have the same resolution')

    # Setup plot.
    plt.figure(figsize=[12, 4.6], dpi=dpi, tight_layout=True)
    hmap_center = hmap_size // 2
    hmap_range = np.arange(hmap_center, hmap_size)
    freqs0 = np.linspace(0, image_size / 2, num=(hmap_size // 2 + 1), endpoint=True)
    freqs45 = np.linspace(0, image_size / np.sqrt(2), num=(hmap_size // 2 + 1), endpoint=True)
    xticks0 = np.linspace(freqs0[0], freqs0[-1], num=9, endpoint=True)
    xticks45 = np.round(np.linspace(freqs45[0], freqs45[-1], num=9, endpoint=True))

    yticks = np.linspace(ymin,ymax, num=9, endpoint=True)

    # Draw 0 degree slice.
    plt.subplot(1, 2, 1)
    plt.title('0\u00b0 slice')
    plt.xlim(xticks0[0], xticks0[-1])
    plt.ylim(yticks[0], yticks[-1])
    plt.xticks(xticks0)
    plt.yticks(yticks)
    for c in cases:
        plt.plot(freqs0, c.hmap[hmap_center, hmap_range], label=c.label)
    plt.grid()
    plt.legend(loc='upper right')

    # Draw 45 degree slice.
    plt.subplot(1, 2, 2)
    plt.title('45\u00b0 slice')
    plt.xlim(xticks45[0], xticks45[-1])
    plt.ylim(yticks[0], yticks[-1])
    plt.xticks(xticks45)
    plt.yticks(yticks)
    for c in cases:
        plt.plot(freqs45, c.hmap[hmap_range, hmap_range], label=c.label)
    plt.grid()
    plt.legend(loc='upper right')

    # Display or save.
    if save is None:
        plt.show()
    else:
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)

#----------------------------------------------------------------------------

@main.command()
@click.argument('npz-files', nargs=-1, required=True)
@click.option('--save', help='Save the plot and exit', metavar='[PNG|PDF|...]')
@click.option('--dpi', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=100, show_default=True)
@click.option('--smooth', help='Amount of smoothing', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--ymax', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=15, show_default=True)
@click.option('--ymin', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=-15, show_default=True)
def slices_std(npz_files, save, dpi, smooth, ymax, ymin):
    """Visualize 1D slices based on the given .npz files."""
    cases = [dnnlib.EasyDict(npz_file=npz_file) for npz_file in npz_files]
    for c in cases:
        c.hmap, c.image_size = construct_heatmap(npz_file=c.npz_file, smooth=smooth)
        c.label = os.path.splitext(os.path.basename(c.npz_file))[0]

        file_name = os.path.basename(c.npz_file)
        std_file = os.path.join(os.path.dirname(c.npz_file) + "_std", file_name)
        c.std_up_hmap, c.std_low_hmap, _ = construct_std_heatmap(mean_npz_file=c.npz_file, npz_file=std_file, smooth=smooth) 
        c.std_up_hmap = c.std_up_hmap - c.hmap
        c.std_low_hmap = c.std_low_hmap - c.hmap
        c.std_hmap = np.concatenate([-c.std_low_hmap[None], c.std_up_hmap[None]], axis=0)

    # Check consistency.
    image_size = cases[0].image_size
    hmap_size = cases[0].hmap.shape[0]
    if any(c.image_size != image_size or c.hmap.shape[0] != hmap_size for c in cases):
        raise click.ClickException('All .npz must have the same resolution')

    # Setup plot.
    plt.figure(figsize=[12, 9.2], dpi=dpi, tight_layout=True)
    hmap_center = hmap_size // 2
    # x_range = np.arange(hmap_center, hmap_size)
    to_angle = lambda x : np.clip((x / 90 * np.pi * 0.5), a_min=1e-8, a_max=np.pi * 0.5 - 1e-8)
    x_range = lambda x : np.linspace(hmap_center, hmap_center + np.clip(hmap_center / np.tan(to_angle(x)), a_max=hmap_center, a_min=0), num=hmap_size//2+1, endpoint=True, dtype=int)
    y_range = lambda x : np.linspace(hmap_center, hmap_center + np.clip(hmap_center * np.tan(to_angle(x)), a_max=hmap_center, a_min=0), num=hmap_size//2+1, dtype=int)
    freqs = lambda x : np.linspace(0, np.sqrt((y_range(x)[-1] / hmap_center - 1) ** 2 + (x_range(x)[-1] / hmap_center - 1) ** 2), num=(hmap_size // 2 + 1), endpoint=True) * image_size / 2
    xticks = lambda x : np.linspace(freqs(x)[0], freqs(x)[-1], num=9, endpoint=True)
    yticks = np.linspace(ymin, ymax, num=9, endpoint=True)

    period = 1
    plot_info_list = [
        # angle, period
        (0, period),
        (15, period),
        (30, period),
        (45, period),
        (60, period), 
        (75, period), 
        (90, period),
    ]

    px = 2
    py = 4

    for i, plot_info in enumerate(plot_info_list):
        i_angle, i_period = plot_info
        i_xticks = xticks(i_angle)
        i_yticks = yticks
        i_freqs = freqs(i_angle)
        i_xrange = x_range(i_angle)
        i_yrange = y_range(i_angle)

        # st_idx = int(len(i_freqs) * 0.8)
        st_idx = 0
        plt.subplot(py, px, i + 1)
        plt.title(f'{i_angle}\u00b0 slice')
        plt.xlim(i_xticks[0], i_xticks[-1])
        plt.ylim(i_yticks[0], i_yticks[-1])
        plt.xticks(i_xticks[0:])
        plt.yticks(i_yticks)

        for c in cases:

            markers, caps, bars = plt.errorbar(i_freqs[st_idx::i_period], c.hmap[i_xrange[st_idx::i_period], i_yrange[st_idx::i_period]], c.std_hmap[:,i_xrange[st_idx::i_period], i_yrange[st_idx::i_period]], label=c.label)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
        plt.grid()
        plt.legend(loc='lower left')

    # Display or save.
    if save is None:
        plt.show()
    else:
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)

#----------------------------------------------------------------------------

@main.command()
@click.argument('npz-files', nargs=-1, required=True)
@click.option('--key', default="image", required=True)
@click.option('--save', help='Save the plot and exit', metavar='[PNG|PDF|...]')
@click.option('--dpi', help='Figure resolution', metavar='FLOAT', type=click.FloatRange(min=1), default=100, show_default=True)
@click.option('--smooth', help='Amount of smoothing', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--ymax', help='Figure resolution', metavar='FLOAT', default=15, show_default=True)
@click.option('--ymin', help='Figure resolution', metavar='FLOAT', default=-15, show_default=True)
def slices_feat(npz_files, key, save, dpi, smooth, ymax, ymin):
    ymax = 0
    ymin = 0
    """Visualize 1D slices based on the given .npz files."""
    cases = [dnnlib.EasyDict(npz_file=npz_file) for npz_file in npz_files]
    for c in cases:
        c.hmap, c.image_size = construct_heatmap(npz_file=c.npz_file, smooth=smooth, key=key)
        c.label = os.path.splitext(os.path.basename(c.npz_file))[0]
        if c.hmap.max() > ymax:
            ymax = c.hmap.max()
        if c.hmap.min() < ymin:
            ymin = c.hmap.min()

        file_name = os.path.basename(c.npz_file)
        std_file = os.path.join(os.path.dirname(c.npz_file) + "_std", file_name)
        c.std_up_hmap, c.std_low_hmap, _ = construct_std_heatmap(mean_npz_file=c.npz_file, npz_file=std_file, smooth=smooth, mean_key=key, std_key=key) 
        c.std_up_hmap = c.std_up_hmap - c.hmap
        c.std_low_hmap = c.std_low_hmap - c.hmap
        c.std_hmap = np.concatenate([-c.std_low_hmap[None], c.std_up_hmap[None]], axis=0)

    # Check consistency.
    image_size = cases[0].image_size
    hmap_size = cases[0].hmap.shape[0]
    if any(c.image_size != image_size or c.hmap.shape[0] != hmap_size for c in cases):
        raise click.ClickException('All .npz must have the same resolution')

    # Setup plot.
    plt.figure(figsize=[12, 9.2], dpi=dpi, tight_layout=True)
    hmap_center = hmap_size // 2
    # x_range = np.arange(hmap_center, hmap_size)
    to_angle = lambda x : np.clip((x / 90 * np.pi * 0.5), a_min=1e-8, a_max=np.pi * 0.5 - 1e-8)
    x_range = lambda x : np.linspace(hmap_center, hmap_center + np.clip(hmap_center / np.tan(to_angle(x)), a_max=hmap_center, a_min=0), num=hmap_size//2+1, endpoint=True, dtype=int)
    y_range = lambda x : np.linspace(hmap_center, hmap_center + np.clip(hmap_center * np.tan(to_angle(x)), a_max=hmap_center, a_min=0), num=hmap_size//2+1, dtype=int)
    freqs = lambda x : np.linspace(0, np.sqrt((y_range(x)[-1] / hmap_center - 1) ** 2 + (x_range(x)[-1] / hmap_center - 1) ** 2), num=(hmap_size // 2 + 1), endpoint=True) * image_size / 2
    xticks = lambda x : np.linspace(freqs(x)[0], freqs(x)[-1], num=9, endpoint=True)
    yticks = np.linspace(ymin, ymax, num=9, endpoint=True)

    period = 1
    plot_info_list = [
        # angle, period
        (0, period),
        (15, period),
        (30, period),
        (45, period),
        (60, period), 
        (75, period), 
        (90, period),
    ]

    px = 2
    py = 4

    for i, plot_info in enumerate(plot_info_list):
        i_angle, i_period = plot_info
        i_xticks = xticks(i_angle)
        i_yticks = yticks
        i_freqs = freqs(i_angle)
        i_xrange = x_range(i_angle)
        i_yrange = y_range(i_angle)

        # st_idx = int(len(i_freqs) * 0.8)
        st_idx = 0
        plt.subplot(py, px, i + 1)
        plt.title(f'{i_angle}\u00b0 slice')
        plt.xlim(i_xticks[0], i_xticks[-1])
        plt.ylim(i_yticks[0], i_yticks[-1])
        plt.xticks(i_xticks[0:])
        plt.yticks(i_yticks)

        for c in cases:

            markers, caps, bars = plt.errorbar(i_freqs[st_idx::i_period], c.hmap[i_xrange[st_idx::i_period], i_yrange[st_idx::i_period]], c.std_hmap[:,i_xrange[st_idx::i_period], i_yrange[st_idx::i_period]], label=c.label)
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
        plt.grid()
        plt.legend(loc='lower left')

    # Display or save.
    if save is None:
        plt.show()
    else:
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------