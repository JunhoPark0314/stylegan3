#PKL_PATH="output/original/00005-stylegan3_stylegan3-t-afhq-v2-128-gpus2-batch16-gamma1/*.pkl"
EXPR_NAME="00003-scalewise_pg_stylegan3-t-spg-afhq-v2-128-gpus2-batch16-gamma8.2"
PKL_PATH="output/scalewise_pg/$EXPR_NAME/*.pkl"
DATA_PATH="data/afhq-v2/afhqv2-15k-128x128.zip"

MAX_NUM="003800"

for file in $PKL_PATH;
do
	pkl_file=${file##*/}
	pkl_num=${pkl_file//[!0-9]/}
	if [[ $((10#$pkl_num)) -le $((10#$MAX_NUM)) ]];then
		echo "${file} PASS"
	else
		echo "python calc_metrics.py --network ${file} --metrics fid50k_full --data $DATA_PATH --mirror 1 --gpus 2"
		python calc_metrics.py --network ${file} --metrics fid50k_full --data $DATA_PATH --mirror 1 --gpus 2
	fi

done