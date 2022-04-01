PKL_PATH="output/original/00005-stylegan3_stylegan3-t-afhq-v2-128-gpus2-batch16-gamma1/*.pkl"
DATA_PATH="data/afhq-v2/afhqv2-15k-128x128.zip"

for file in $PKL_PATH;
do
	if [[ "${file}" == *"000000"* ]];then
		echo "${file} PASS"
	else
		echo "python calc_metrics.py --network ${file} --metrics fid50k_full,pr50k3_full --data $DATA_PATH --mirror 1 --gpus 2"
		python calc_metrics.py --network ${file} --metrics fid50k_full,pr50k3_full --data $DATA_PATH --mirror 1 --gpus 2
	fi

done
