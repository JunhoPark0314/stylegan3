#PKL_PATH="output/original/00005-stylegan3_stylegan3-t-afhq-v2-128-gpus2-batch16-gamma1/*.pkl"
# EXPR_NAME="00003-scalewise_pg_stylegan3-t-spg-afhq-v2-128-gpus2-batch16-gamma8.2"

PKL_PATH="output/ablation/**/*afhqv2-cat-5k*/*.pkl"
DATA_PATH="fast_data/afhqv2-cat-5k-128x128.zip"
METRICS="fid50k_full,fid50k_full_64,fid50k_full_32,fid50k_full_crop_32"

# MAX_NUM="003800"
MAX_NUM = "000000"
CNT=0

for file in $PKL_PATH;
do
	pkl_file=${file##*/}
	pkl_num=${pkl_file//[!0-9]/}
	if [[ $((10#$pkl_num)) -le $((10#$MAX_NUM)) ]];then
		echo "${file} PASS"
	else
		echo "python calc_metrics.py --network ${file} --metrics ${METRICS} --data $DATA_PATH --mirror 1 --gpus 2"
		python calc_metrics.py --network ${file} --metrics ${METRICS} --data $DATA_PATH --mirror 1 --gpus 2
		CNT=$(( CNT + 1 ))
	fi
done

echo "$CNT items dones"