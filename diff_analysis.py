import click
import torch
import dnnlib
import legacy
import numpy as np
from training import dataset
import os
from torchvision.utils import save_image
import tqdm
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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

def construct_heatmap(spectrum, smooth, image_size):
	spectrum = np.clip(spectrum, a_min=1e-50, a_max=1e8)
	hmap = np.log10(spectrum) * 10 # dB
	hmap = np.fft.fftshift(hmap)
	hmap = np.concatenate([hmap, hmap[:1, :]], axis=0)
	hmap = np.concatenate([hmap, hmap[:, :1]], axis=1)
	if smooth > 0:
		sigma = spectrum.shape[0] / image_size * smooth
		hmap = gaussian_filter(hmap, sigma=sigma, mode='nearest')
	return hmap, image_size

prev_dir = "output/diff_test/rand_train_1000/100_iter_diff_test/005008"
curr_dir = "output/diff_test/rand_train_1000/100_iter_diff_test/006608"
dest = "output/diff_test/rand_train_1000/100_iter_diff_test/spectrum_diff.npz"
device = torch.device("cuda")

num_images, image_size, prev_iter_image = stream_source_images(prev_dir, None, 0, device)
num_images, image_size, curr_iter_image = stream_source_images(curr_dir, None, 0, device)

interp = 4
beta = 8
spectrum_size = image_size * interp
padding = spectrum_size - image_size
window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)
window *= window.square().sum().rsqrt()
window = window.ger(window).unsqueeze(0).unsqueeze(1)

mean_spectrum = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)
for prev_image, curr_image in tqdm.tqdm(zip(prev_iter_image, curr_iter_image), total=num_images):
	prev_image = prev_image / 127.5 - 1
	curr_image = curr_image / 127.5 - 1 
	
	prev_image = torch.nn.functional.pad(prev_image * window, [0, padding, 0, padding])
	curr_image = torch.nn.functional.pad(curr_image * window, [0, padding, 0, padding])
	image_diff = (prev_image - curr_image)
	mean_spectrum += torch.fft.fftn(image_diff, dim=[2,3]).abs().square().mean(dim=[0,1])

mean_spectrum /= num_images

num_images, image_size, prev_iter_image = stream_source_images(prev_dir, None, 0, device)
num_images, image_size, curr_iter_image = stream_source_images(curr_dir, None, 0, device)

std_spectrum = torch.zeros([spectrum_size, spectrum_size], dtype=torch.float64, device=device)
for prev_image, curr_image in tqdm.tqdm(zip(prev_iter_image, curr_iter_image), total=num_images):
	prev_image = prev_image / 127.5 - 1
	curr_image = curr_image / 127.5 - 1 
	
	prev_image = torch.nn.functional.pad(prev_image * window, [0, padding, 0, padding])
	curr_image = torch.nn.functional.pad(curr_image * window, [0, padding, 0, padding])
	image_diff = (prev_image - curr_image)
	diff_spectrum = torch.fft.fftn(image_diff, dim=[2,3]).abs().square().mean(dim=[0,1])
	std_spectrum += (diff_spectrum - mean_spectrum).square()

std_spectrum /= num_images

if os.path.dirname(dest):
	os.makedirs(os.path.dirname(dest), exist_ok=True)

np.savez(dest, mean_spectrum=mean_spectrum.cpu().numpy(), std_spectrum=std_spectrum.cpu().numpy(), image_size=image_size)

dir_name = os.path.dirname(dest)

mean_hmap, image_size = construct_heatmap(mean_spectrum.cpu().numpy(), smooth=0.25, image_size=image_size)
up_hmap, image_size = construct_heatmap(mean_spectrum.cpu().numpy() + std_spectrum.sqrt().cpu().numpy(), smooth=0.25, image_size=image_size)
down_hmap, image_size = construct_heatmap(mean_spectrum.cpu().numpy() - std_spectrum.sqrt().cpu().numpy(), smooth=0.25, image_size=image_size)
std_hmap = np.concatenate([(mean_hmap - down_hmap).reshape(1,513,513), (up_hmap - mean_hmap).reshape(1, 513, 513)])


dpi = 300
hmap_size = mean_hmap.shape[0]
ymin=max(down_hmap.min(), -40)
ymax=up_hmap.max()

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

	markers, caps, bars = plt.errorbar(i_freqs[st_idx::i_period], mean_hmap[i_xrange[st_idx::i_period], i_yrange[st_idx::i_period]], std_hmap[:,i_xrange[st_idx::i_period], i_yrange[st_idx::i_period]], label='test')
	[bar.set_alpha(0.5) for bar in bars]
	[cap.set_alpha(0.5) for cap in caps]

	plt.grid()
	plt.legend(loc='lower left')

# Display or save.
if os.path.dirname(dest):
	os.makedirs(os.path.dirname(dest), exist_ok=True)

plt.savefig(os.path.join(os.path.dirname(dest), 'fft_plot.png'))