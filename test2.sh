FILE="output/ablation/ours/00001-stylegan2-fdpk-blur-afhqv2-cat-5k-128x128-gpus2-batch32-gamma0.5-freq_dist:uniform-fdim_max512-fdim_base8-sort_distTrue/*.pkl"

for file in $FILE;
do
	python discriminator_test.py calc-mag --source $file
done
	