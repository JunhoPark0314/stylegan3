# EXPR_NAME="Original-0.5"
# MIN_NUM="000000"

# EXPR_NAME="Original-blur-0.5"
# MIN_NUM="008400"

# EXPR_NAME="33:SPG-blur-afhqv2-cat-256"
# MIN_NUM="006000"

# EXPR_NAME="15:Original-afhqv2-cat-256"
# MIN_NUM="003800"

EXPR_NAME="34:Original-blur-afhqv2-cat-256"
MIN_NUM="001200"

PKL_PATH="output/Results/$EXPR_NAME/*400.pkl"

TRAINSET="afhqv2-cat-5k-256x256"


for file in $PKL_PATH;
do

	pkl_file=${file##*/}
	pkl_num=${pkl_file//[!0-9]/}
	if [[ $((10#$pkl_num)) -le $((10#$MIN_NUM)) ]];then
		echo "${file} PASS"
	else
		echo "python avg_spectra.py calc --source ${file} --dest output/Results/${EXPR_NAME}/NPZ/${pkl_num}.npz --mean 127.5 --std 127.5 --num 15000"
		# python avg_spectra.py calc --source ${file} --dest output/Results/${EXPR_NAME}/NPZ/${pkl_num}.npz --mean 127.5 --std 127.5 --num 15000
		echo "python avg_spectra.py calc-std --source ${file} --dest output/Results/${EXPR_NAME}/NPZ_std/${pkl_num}.npz  --npz_mean output/Results/${EXPR_NAME}/NPZ/${pkl_num}.npz\
			--mean 127.5 --std 127.5 --num 15000"
		# python avg_spectra.py calc-std --source ${file} --dest output/Results/${EXPR_NAME}/NPZ_std/${pkl_num}.npz  --npz_mean output/Results/${EXPR_NAME}/NPZ/${pkl_num}.npz \
		# 	--mean 127.5 --std 127.5 --num 15000
		echo "python avg_spectra.py slices output/Results/Trainset/NPZ/${TRAINSET}.npz output/Results/${EXPR_NAME}/NPZ/${pkl_num}.npz --save output/Results/${EXPR_NAME}/Slices/${pkl_num}.png --dpi 300"
		#python avg_spectra.py slices output/Results/Trainset/NPZ/training-data.npz output/Results/${EXPR_NAME}/NPZ/${pkl_num}.npz --save output/Results/${EXPR_NAME}/Slices/${pkl_num}.png --dpi 300
		python avg_spectra.py slices-std output/Results/Trainset/NPZ/${TRAINSET}.npz output/Results/${EXPR_NAME}/NPZ/${pkl_num}.npz --save output/Results/${EXPR_NAME}/Slices_std/${pkl_num}.png --dpi 300
	fi

done