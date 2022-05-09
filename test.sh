FILE="output/ablation/baseline/00000-stylegan2-afhqv2-cat-5k-128x128-gpus2-batch32-gamma0.5/*.pkl"

for file in $FILE;
do
	python discriminator_test.py calc-mag --source $file
done
	