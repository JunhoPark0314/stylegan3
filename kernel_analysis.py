from collections import defaultdict
from glob import glob
import os
import re
import torch
import legacy
import dnnlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

force_plot = True

def draw_pca(D, conv0, conv1, label):
	len_res = len(D.block_resolutions)
	conv0_kernel = dict()
	conv1_kernel = dict()
	for i, res in enumerate(D.block_resolutions):
		block = getattr(D, f'b{res}')
		conv0_weight = conv0(block).flatten(0, 1).flatten(1,2)
		conv1_weight = conv1(block).flatten(0, 1).flatten(1,2)

		# conv0_weight *= conv0_weight.square().mean(dim=[1,2], keepdim=True).rsqrt()
		# _, conv0_s, _ = torch.pca_lowrank(conv0_weight.flatten(1,2), q=9, niter=30)
		# conv0_s /= conv0_s.max()

		# conv1_weight *= conv1_weight.square().mean(dim=[1,2], keepdim=True).rsqrt()
		# _, conv1_s, _ = torch.pca_lowrank(conv1_weight.flatten(1,2), q=9, niter=30)
		# conv1_s /= conv1_s.max()

		conv0_kernel[res] = conv0_weight
		conv1_kernel[res] = conv1_weight

		# plt.subplot(2,len_res,i+1)
		# plt.plot(conv0_s, label=label)
		# plt.legend()
		# plt.subplot(2,len_res,i+1+len_res)
		# plt.plot(conv1_s, label=label)
		# plt.legend()
	return conv0_kernel, conv1_kernel

def draw_norm(D, conv0, conv1, label):
	len_res = len(D.block_resolutions)
	for i, res in enumerate(D.block_resolutions):
		block = getattr(D, f'b{res}')
		conv0_weight = conv0(block)
		conv1_weight = conv1(block)

		conv0_norm = conv0_weight.flatten(0,1).flatten(1,2).norm(dim=-1)
		conv1_norm = conv1_weight.flatten(0,1).flatten(1,2).norm(dim=-1)

		conv0_norm =torch.sort(conv0_norm)[0]
		conv1_norm =torch.sort(conv1_norm)[0]

		plt.subplot(2,len_res,i+1)
		plt.plot(conv0_norm,'.', label=label)
		plt.legend()
		plt.subplot(2,len_res,i+1+len_res)
		plt.plot(conv1_norm,'.', label=label)
		plt.legend()

def draw_corr(D, conv0, conv1, label):
	len_res = len(D.block_resolutions)
	for i, res in enumerate(D.block_resolutions):
		block = getattr(D, f'b{res}')
		conv0_weight = conv0(block)
		conv1_weight = conv1(block)

		conv0_s = []
		for j in range(conv0_weight.shape[1]):
			curr_weight = conv0_weight[:,j,...]
			curr_weight *= curr_weight.square().sum(dim=[1,2], keepdim=True).rsqrt()
			sim = (curr_weight.flatten(1,2) @ curr_weight.flatten(1,2).T) 
			sim = sim / sim.abs().max()
			conv0_s.append(sim.unsqueeze(0))
			if j > 48:
				break
		conv0_s = torch.cat(conv0_s, dim=0)

		conv1_s = []
		for j in range(conv1_weight.shape[1]):
			curr_weight = conv1_weight[:,j,...]
			curr_weight *= curr_weight.square().sum(dim=[1,2], keepdim=True).rsqrt()
			sim = (curr_weight.flatten(1,2) @ curr_weight.flatten(1,2).T) 
			sim = sim / sim.abs().max()
			conv1_s.append(sim.unsqueeze(0))
			if j > 48:
				break
		conv1_s = torch.cat(conv1_s, dim=0)

		num_h = 10
		len_h0 = len(conv0_s)
		len_h1 = len(conv1_s)

		s_0 = conv0_s.shape[-1]
		s_1 = conv1_s.shape[-1]
		conv0_s = torch.nn.functional.pad(conv0_s.reshape(len_h0//num_h, num_h, s_0, s_0), [4]*4)
		conv1_s = torch.nn.functional.pad(conv1_s.reshape(len_h1//num_h, num_h, s_1, s_1), [4]*4)

		conv0_s = conv0_s.permute(0, 2, 1, 3).flatten(0,1).flatten(1,2)
		conv1_s = conv1_s.permute(0, 2, 1, 3).flatten(0,1).flatten(1,2)

		conv0_s = torch.einsum('hwr,rc->chw', conv0_s.relu().unsqueeze(-1), torch.tensor([[1,0,0]])) +\
				 torch.einsum('hwr,rc->chw', (-conv0_s).relu().unsqueeze(-1), torch.tensor([[0,0,1]]))
	
		conv1_s = torch.einsum('hwr,rc->chw', conv1_s.relu().unsqueeze(-1), torch.tensor([[1,0,0]])) +\
				 torch.einsum('hwr,rc->chw', (-conv1_s).relu().unsqueeze(-1), torch.tensor([[0,0,1]]))
		
		save_image(conv0_s, f'kernel_analysis/{label}_{res}_corr_conv0.png')
		save_image(conv1_s, f'kernel_analysis/{label}_{res}_corr_conv1.png')

def draw_basis_shift_sim_orig(D_list, conv0, conv1, label):
	per_res_diff = defaultdict(list)
	plot = True
	if os.path.isdir(os.path.dirname(f'output/images/{label}/_sim.png')):
		plot = force_plot
	orig_D = D_list[0]
	for j, (prev_D, curr_D)  in enumerate(zip(D_list[:-1], D_list[1:])):
		for _, res in enumerate(prev_D.block_resolutions):
			prev_block = getattr(orig_D, f'b{res}')
			curr_block = getattr(curr_D, f'b{res}')

			prev_conv0_basis = conv0(prev_block).flatten(2,3)
			prev_conv1_basis = conv1(prev_block).flatten(2,3)
			curr_conv0_basis = conv0(curr_block).flatten(2,3)
			curr_conv1_basis = conv1(curr_block).flatten(2,3)

			prev_conv0_basis = (prev_conv0_basis / prev_conv0_basis.norm(dim=-1, keepdim=True))
			prev_conv1_basis = (prev_conv1_basis / prev_conv1_basis.norm(dim=-1, keepdim=True))
			curr_conv0_basis = (curr_conv0_basis / curr_conv0_basis.norm(dim=-1, keepdim=True))
			curr_conv1_basis = (curr_conv1_basis / curr_conv1_basis.norm(dim=-1, keepdim=True))

			conv0_sim = (curr_conv0_basis * prev_conv0_basis).sum(dim=-1).abs()
			conv1_sim = (curr_conv1_basis * prev_conv1_basis).sum(dim=-1).abs()

			# save_image(conv0_sim, f'{label}_{j}_conv0_diff.png')
			# save_image(conv1_sim, f'{label}_{j}_conv0_diff.png')

			per_res_diff[f'{res}_conv0'].append(conv0_sim.unsqueeze(0))
			per_res_diff[f'{res}_conv1'].append(conv1_sim.unsqueeze(0))
	
	for k, v in per_res_diff.items():
		per_res_diff[k] = 1 - torch.cat(per_res_diff[k], dim=0)
		if plot:
			os.makedirs(os.path.dirname(f'output/images/{label}/{k}_sim.png'), exist_ok=True)
			save_image(torch.nn.functional.pad(per_res_diff[k], [8] * 4).flatten(0,1) / per_res_diff[k].max(), f'output/images/{label}/{k}_sim.png')

	return per_res_diff

def draw_basis_shift_sim(D_list, conv0, conv1, label):
	per_res_diff = defaultdict(list)
	plot = True
	if os.path.isdir(os.path.dirname(f'output/images/{label}/_sim.png')):
		plot = force_plot
	for j, (prev_D, curr_D)  in enumerate(zip(D_list[:-1], D_list[1:])):
		for _, res in enumerate(prev_D.block_resolutions):
			prev_block = getattr(prev_D, f'b{res}')
			curr_block = getattr(curr_D, f'b{res}')

			prev_conv0_basis = conv0(prev_block).flatten(2,3)
			prev_conv1_basis = conv1(prev_block).flatten(2,3)
			curr_conv0_basis = conv0(curr_block).flatten(2,3)
			curr_conv1_basis = conv1(curr_block).flatten(2,3)

			prev_conv0_basis = (prev_conv0_basis / prev_conv0_basis.norm(dim=-1, keepdim=True))
			prev_conv1_basis = (prev_conv1_basis / prev_conv1_basis.norm(dim=-1, keepdim=True))
			curr_conv0_basis = (curr_conv0_basis / curr_conv0_basis.norm(dim=-1, keepdim=True))
			curr_conv1_basis = (curr_conv1_basis / curr_conv1_basis.norm(dim=-1, keepdim=True))

			conv0_sim = (curr_conv0_basis * prev_conv0_basis).sum(dim=-1).abs()
			conv1_sim = (curr_conv1_basis * prev_conv1_basis).sum(dim=-1).abs()

			# save_image(conv0_sim, f'{label}_{j}_conv0_diff.png')
			# save_image(conv1_sim, f'{label}_{j}_conv0_diff.png')

			per_res_diff[f'{res}_conv0'].append(conv0_sim.unsqueeze(0))
			per_res_diff[f'{res}_conv1'].append(conv1_sim.unsqueeze(0))
	
	for k, v in per_res_diff.items():
		per_res_diff[k] = 1 - torch.cat(per_res_diff[k], dim=0)
		if plot:
			os.makedirs(os.path.dirname(f'output/images/{label}/{k}_sim.png'), exist_ok=True)
			save_image(torch.nn.functional.pad(per_res_diff[k], [8] * 4).flatten(0,1) / per_res_diff[k].max(), f'output/images/{label}/{k}_sim.png')

	return per_res_diff

def draw_basis_shift_norm(D_list, conv0, conv1, label):
	per_res_diff = defaultdict(list)
	plot = True
	if os.path.isdir(os.path.dirname(f'output/images/{label}/_sim.png')):
		plot = force_plot
	for j, (prev_D, curr_D)  in enumerate(zip(D_list[:-1], D_list[1:])):
		for _, res in enumerate(prev_D.block_resolutions):
			prev_block = getattr(prev_D, f'b{res}')
			curr_block = getattr(curr_D, f'b{res}')

			prev_conv0_basis = conv0(prev_block).flatten(2,3)
			prev_conv1_basis = conv1(prev_block).flatten(2,3)
			curr_conv0_basis = conv0(curr_block).flatten(2,3)
			curr_conv1_basis = conv1(curr_block).flatten(2,3)

			conv0_diff = (curr_conv0_basis.norm(dim=-1) - prev_conv0_basis.norm(dim=-1)).abs()
			conv1_diff = (curr_conv1_basis.norm(dim=-1) - prev_conv1_basis.norm(dim=-1)).abs()

			# save_image(conv0_sim, f'{label}_{j}_conv0_diff.png')
			# save_image(conv1_sim, f'{label}_{j}_conv0_diff.png')

			per_res_diff[f'{res}_conv0'].append(conv0_diff.unsqueeze(0))
			per_res_diff[f'{res}_conv1'].append(conv1_diff.unsqueeze(0))
	
	for k, v in per_res_diff.items():
		per_res_diff[k] = torch.cat(per_res_diff[k], dim=0)
		if plot:
			os.makedirs(os.path.dirname(f'output/images/{label}/{k}_norm.png'), exist_ok=True)
			save_image(torch.nn.functional.pad(per_res_diff[k], [8] * 4).flatten(0,1) / per_res_diff[k].max(), f'output/images/{label}/{k}_norm.png')
			for num_iter in range(len(per_res_diff[k])):
				plt.plot()

	return per_res_diff

def draw_basis_shift_diff(D_list, conv0, conv1, label):
	plot = True
	if os.path.isdir(os.path.dirname(f'output/images/{label}/_sim.png')):
		plot = force_plot
	per_res_diff = defaultdict(list)
	for j, (prev_D, curr_D)  in enumerate(zip(D_list[:-1], D_list[1:])):
		for _, res in enumerate(prev_D.block_resolutions):
			prev_block = getattr(prev_D, f'b{res}')
			curr_block = getattr(curr_D, f'b{res}')

			prev_conv0_basis = conv0(prev_block).flatten(2,3)
			prev_conv1_basis = conv1(prev_block).flatten(2,3)
			curr_conv0_basis = conv0(curr_block).flatten(2,3)
			curr_conv1_basis = conv1(curr_block).flatten(2,3)

			conv0_diff = (curr_conv0_basis - prev_conv0_basis).norm(dim=-1)
			conv1_diff = (curr_conv1_basis - prev_conv1_basis).norm(dim=-1)

			# save_image(conv0_sim, f'{label}_{j}_conv0_diff.png')
			# save_image(conv1_sim, f'{label}_{j}_conv0_diff.png')

			per_res_diff[f'{res}_conv0'].append(conv0_diff.unsqueeze(0))
			per_res_diff[f'{res}_conv1'].append(conv1_diff.unsqueeze(0))
	
	for k, v in per_res_diff.items():
		per_res_diff[k] = torch.cat(per_res_diff[k], dim=0)
		if plot:
			os.makedirs(os.path.dirname(f'output/images/{label}/{k}_diff.png'), exist_ok=True)
			save_image(torch.nn.functional.pad(per_res_diff[k], [8] * 4).flatten(0,1) / per_res_diff[k].max(), f'output/images/{label}/{k}_diff.png')

	return per_res_diff

def draw_overall_diversity(D_list, conv0, conv1, label):
	per_res_diff = defaultdict(list)
	len_res = len(D_list[0].block_resolutions)
	for j, curr_D  in enumerate(D_list):
		for _, res in enumerate(curr_D.block_resolutions):
			curr_block = getattr(curr_D, f'b{res}')

			curr_conv0_basis = conv0(curr_block).flatten(2,3)
			curr_conv1_basis = conv1(curr_block).flatten(2,3)

			# save_image(conv0_sim, f'{label}_{j}_conv0_diff.png')
			# save_image(conv1_sim, f'{label}_{j}_conv0_diff.png')

			per_res_diff[f'{res}_conv0'].append(curr_conv0_basis.flatten(0,1))
			per_res_diff[f'{res}_conv1'].append(curr_conv1_basis.flatten(0,1))
	
	i=0
	for k, v in per_res_diff.items():
		per_res_diff[k] = torch.cat(per_res_diff[k], dim=0)
		_, s, _ = torch.pca_lowrank(per_res_diff[k], q=9, niter=50)
		s /= s.max()
		plt.subplot(1, len_res*2, i+1)
		plt.plot(s, label=label)
		plt.legend()
		i+=1
		# save_image(torch.nn.functional.pad(per_res_diff[k], [8] * 4).flatten(0,1) / per_res_diff[k].max(), f'{label}_{k}_.png')

	return per_res_diff


def draw_conv_norm(D_baseline, D_ours, conv0_baseline, conv1_baseline, conv0_ours, conv1_ours):
	for i, res in enumerate(D_ours.block_resolutions):
		block_ours = getattr(D_ours, f'b{res}')
		block_baseline = getattr(D_baseline, f'b{res}')

		# conv0_weight_ours = torch.sort(conv0_ours(block_ours).norm(dim=[2,3]).flatten())[0]
		# conv1_weight_ours = torch.sort(conv1_ours(block_ours).norm(dim=[2,3]).flatten())[0]


		conv0_weight_baseline = torch.sort(conv0_baseline(block_baseline).norm(dim=[2,3]).flatten())[0]
		conv1_weight_baseline = torch.sort(conv1_baseline(block_baseline).norm(dim=[2,3]).flatten())[0]

		plt.clf()
		# plt.plot(conv0_weight_baseline, label="baseline")
		plt.plot(conv0_weight_ours, label="ours")
		plt.savefig(f'conv0_{res}.png')

		plt.clf()
		# plt.plot(conv1_weight_baseline, label="baseline")
		plt.plot(conv1_weight_ours, label="ours")
		plt.savefig(f'conv1_{res}.png')
	
conv0_baseline = lambda x : x.conv0.weight * x.conv0.weight_gain
conv1_baseline = lambda x : x.conv1.weight * x.conv1.weight_gain
conv0_ours = lambda x : x.conv0_weight(torch.device("cpu"))
conv1_ours = lambda x : x.conv1_weight(torch.device("cpu"))

# conv0_ours = lambda x : x.conv0.weight_gen(torch.ones([1,3,3,3]))
# conv1_ours = lambda x : x.conv1.weight_gen(torch.ones([1,3,3,3]))

def load_network(network_list):
	D_list = []
	pkl_format = network_list[0]
	pkl_list = sorted(glob(pkl_format), key= lambda x : re.findall('[0-9]+', os.path.basename(x))[0])
	for pkl in pkl_list:
		try:
			with dnnlib.util.open_url(pkl) as f:
				D_list.append(legacy.load_network_pkl(f)['D'])
		except:
			print("last_pkl: ", pkl)
			break
	return D_list

baseline_2x_list = [
	"output/diversity/baseline_2x/network-snapshot-{pid:06d}.pkl",
]

baseline_wi_mag_list = [
	"output/diversity/baseline_wi_mag/network-snapshot-{pid:06d}.pkl",
]

baseline_4x_list = [
	"output/diversity/baseline_4x/network-snapshot-{pid:06d}.pkl",
]

mag_fix_list = [
	"output/diversity/mag_fix/network-snapshot-000000.pkl",
	"output/diversity/mag_fix/network-snapshot-001000.pkl",
	"output/diversity/mag_fix/network-snapshot-002000.pkl",
	"output/diversity/mag_fix/network-snapshot-003000.pkl",
]

mixing_fix_list = [
	"output/diversity/mixing_fix/network-snapshot-000000.pkl",
	"output/diversity/mixing_fix/network-snapshot-001000.pkl",
	"output/diversity/mixing_fix/network-snapshot-002000.pkl",
]

mag_mixing_fix_list = [
	"output/diversity/mag_mixing_fix/network-snapshot-000000.pkl",
	"output/diversity/mag_mixing_fix/network-snapshot-001000.pkl",
	"output/diversity/mag_mixing_fix/network-snapshot-002000.pkl",
	"output/diversity/mag_mixing_fix/network-snapshot-003000.pkl",
	"output/diversity/mag_mixing_fix/network-snapshot-004000.pkl",
	"output/diversity/mag_mixing_fix/network-snapshot-005000.pkl",
]

baseline_list = [
	"output/diversity/00000-stylegan2-afhqv2-cat-5k-32x32-gpus2-batch64-gamma0-cbase16384-cmax512/network-snapshot-*.pkl",
]

baseline_noreg_list = [
	"output/diversity/00006-stylegan2-afhqv2-cat-5k-128x128-gpus2-batch32-gamma0-cbase16384-cmax256-no_r1/network-snapshot-{pid:06d}.pkl",
]

random_train_list = [
	# "output/diversity/random_train/network-snapshot-{pid:06d}.pkl",
	"output/diversity/00026-stylegan2-fdpk-afhqv2-cat-5k-512x512-gpus4-batch64-gamma0.5-cbase32768-cmax512-freq_dist:random_train-fdim_max512-fdim_base8-sort_distTrue/network-snapshot-{pid:06d}.pkl",
]

baseline_1iter_train_list = [
	"output/diff_test/baseline_1000/1_iter_diff_test/network-snapshot-005008.pkl",
	"output/diff_test/baseline_1000/1_iter_diff_test/network-snapshot-005024.pkl",
]

rand_1iter_train_list = [
	"output/diff_test/rand_train_1000/1_iter_diff_test/network-snapshot-005008.pkl",
	"output/diff_test/rand_train_1000/1_iter_diff_test/network-snapshot-005024.pkl",
]

baseline_no_reg_list = [
	"output/diversity/baseline_no_reg/network-snapshot-000000.pkl",
	"output/diversity/baseline_no_reg/network-snapshot-001000.pkl",
	"output/diversity/baseline_no_reg/network-snapshot-002000.pkl",
	"output/diversity/baseline_no_reg/network-snapshot-003000.pkl",
	"output/diversity/baseline_no_reg/network-snapshot-004000.pkl",
	"output/diversity/baseline_no_reg/network-snapshot-005000.pkl",
]

baseline_ttur_list = [
	"output/diversity/baseline_ttur/network-snapshot-000000.pkl",
	"output/diversity/baseline_ttur/network-snapshot-001000.pkl",
	"output/diversity/baseline_ttur/network-snapshot-002000.pkl",
]

# plt.figure(figsize=(40,10))
# draw_pca(D_baseline, conv0_baseline, conv1_baseline, "baseline")
# plt.savefig("pca.png")
# draw_pca(D_ours, conv0_ours, conv1_ours, "ours")

# draw_corr(D_baseline, conv0_baseline, conv1_baseline, "baseline")
# draw_corr(D_random_train, conv0_ours, conv1_ours, "4000/random_train_004000")
# draw_corr(D_four_train, conv0_ours, conv1_ours, "four_train")
# draw_corr(D_random_fixed, conv0_ours, conv1_ours, "random_fixed")
# draw_corr(D_random_train_iof, conv0_ours, conv1_ours, "random_train_iof")

plt.figure(figsize=(40,10))
# draw_norm(D_baseline, conv0_baseline, conv1_baseline, "baseline")
# draw_norm(D_random_train, conv0_ours, conv1_ours, "ours")

conv0_weight_baseline = lambda x : x.conv0.weight
conv1_weight_baseline = lambda x : x.conv1.weight

# conv0_weight_ours = lambda x : x.conv0_weight.basis.permute(0,3,1,2)
# conv1_weight_ours = lambda x : x.conv0_weight.basis.permute(0,3,1,2)
conv0_weight_ours = lambda x : x.conv0_weight.basis.permute(0,3,1,2)
conv1_weight_ours = lambda x : x.conv0_weight.basis.permute(0,3,1,2)

target_networks = [
	# (random_train_list, (conv0_ours, conv1_ours), "random_train"), 
	# (baseline_noreg_list, (conv0_baseline, conv1_baseline), "baseline_noreg"), 
	(baseline_list, (conv0_baseline, conv1_baseline), "baseline"),
	# (baseline_2x_list, (conv0_baseline, conv1_baseline), "baseline_2x"),
	# (baseline_4x_list, (conv0_baseline, conv1_baseline), "baseline_4x"),
	# (baseline_ttur_list, (conv0_baseline, conv1_baseline), "baseline_ttur"),

	# (baseline_wi_mag_list, (conv0_baseline, conv1_baseline), "baseline_wi_mag"),
	# (mag_mixing_fix_list, (conv0_ours, conv1_ours), "mag_mixing_train"), 
	# (mag_fix_list, (conv0_ours, conv1_ours), "mag_fix"), 
	# (mixing_fix_list, (conv0_ours, conv1_ours), "mixing_fix"),

	# (baseline_no_reg_list, (conv0_baseline, conv1_baseline), "baseline_no_reg"),
	# (baseline_1iter_train_list, (conv0_baseline, conv1_baseline), "baseline_1iter"),
	# (rand_1iter_train_list, (conv0_ours, conv1_ours), "random_1iter_train"), 
]

sim_results = {}
sim_orig_results = {}
diff_results = {}
norm_results = {}

for D_name_list, conv_keys, label in target_networks:
	D_list = load_network(D_name_list)
	print("pa")
	sim_results[label] = draw_basis_shift_sim(
		D_list, *conv_keys, label
	)
	sim_orig_results[label] = draw_basis_shift_sim_orig(
		D_list, *conv_keys, label
	)
	diff_results[label] = draw_basis_shift_diff(
		D_list, *conv_keys, label
	)
	norm_results[label] = draw_basis_shift_norm(
		D_list, *conv_keys, label
	)
	# conv0_list = defaultdict(list)
	# conv1_list = defaultdict(list)
	# for i, D_sample in enumerate(D_list):
	# 	conv0_kernel, conv1_kernel  = draw_pca(D_sample, *conv_keys, label + f'_{i}')
	# 	for k in conv0_kernel.keys():
	# 		conv0_list[k].append(conv0_kernel[k])
	# 		conv1_list[k].append(conv1_kernel[k])
		
	continue

	# len_res = len(conv0_list)
	len_res = 1
	for j, k in enumerate(list(conv0_list.keys())):
		if k != 128:
			continue
		conv0_list[k] = torch.cat(conv0_list[k])
		conv1_list[k] = torch.cat(conv1_list[k])

		_, conv0_s, _ = torch.pca_lowrank(torch.fft.rfft2(conv0_list[k]).abs().flatten(1,), q=9, niter=10)
		_, conv1_s, _ = torch.pca_lowrank(torch.fft.rfft2(conv1_list[k]).abs().flatten(1,), q=9, niter=10)
		# _, conv0_s, _ = torch.linalg.svd(conv0_list[k])
		# _, conv1_s, _ = torch.linalg.svd(conv1_list[k])

		# conv0_s /= conv0_s.max()
		# conv1_s /= conv1_s.max()

		plt.subplot(2,len_res,j+1)
		plt.plot(conv0_s, label=label)
		plt.legend()
		plt.subplot(2,len_res,j+1+len_res)
		plt.plot(conv1_s, label=label)
		plt.legend()

# os.makedirs(os.path.dirname(f"output/images/pca/pca.png"), exist_ok=True)
# plt.savefig(f"output/images/pca/pca.png")
# print(f"plot  output/images/pca/pca.png")
# plt.clf()

for k in list(sim_results.keys()):
	print("--------------------------------")
	print(k)
	for res in (sim_results[k].keys()):
		print(res)
		print(f"{k} sim: ", sim_results[k][res].mean(dim=[1,2]))
		print(f"{k} sim orig: ", sim_orig_results[k][res].mean(dim=[1,2]))
		print(f"{k} diff: ", diff_results[k][res].mean(dim=[1,2]))
		print(f"{k} norm: ", norm_results[k][res].mean(dim=[1,2]))


# ours_shift = draw_overall_diversity(
# 	[D_random_train_0000, D_random_train_1000, D_random_train_2000, D_random_train_3000, D_random_train_4000, D_random_train_5000],
# 	conv0_ours, conv1_ours,
# 	"ours"
# )

# basis_shift = draw_overall_diversity(
# 	[D_baseline_0000, D_baseline_1000, D_baseline_2000, D_baseline_3000, D_baseline_4000, D_baseline_5000],
# 	conv0_baseline, conv1_baseline,
# 	"baseline"
# )

# basis_wi_mag_shift = draw_overall_diversity(
# 	[D_baseline_wi_mag_0000, D_baseline_wi_mag_1000, D_baseline_wi_mag_2000, D_baseline_wi_mag_3000],
# 	conv0_baseline, conv1_baseline,
# 	"baseline_wi_mag"
# )

# basis_2x_shift = draw_overall_diversity(
# 	[D_baseline_2x_0000, D_baseline_2x_1000, D_baseline_2x_2000, D_baseline_2x_3000],
# 	conv0_baseline, conv1_baseline,
# 	"baseline_2x"
# )


# draw_conv_norm(
# 	D_baseline_5000,
# 	D_random_train_5000,
# 	conv0_baseline, conv1_baseline,
# 	conv0_weight_ours, conv1_weight_ours,
# )



print("done")

# plt.tight_layout()
# plt.savefig("test.png")