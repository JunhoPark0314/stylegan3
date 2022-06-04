NPZ_FILE="output/ablation/baseline/00000-stylegan2-afhqv2-cat-5k-128x128-gpus2-batch32-gamma0.5/NPZ_FEAT/005000.npz"
NPZ_FILE2="output/ablation/ours/00001-stylegan2-fdpk-blur-afhqv2-cat-5k-128x128-gpus2-batch32-gamma0.5-freq_dist:uniform-fdim_max512-fdim_base8-sort_distTrue/NPZ_FEAT/005001.npz"
SAVE="output/ablation/baseline/00000-stylegan2-afhqv2-cat-5k-128x128-gpus2-batch32-gamma0.5/FEAT_slices/005000_"
python avg_spectra.py slices-feat $NPZ_FILE $NPZ_FILE2 --key image --save "$SAVE-image.png" --dpi 300 --ymax 15 --ymin -15
python avg_spectra.py slices-feat $NPZ_FILE $NPZ_FILE2 --key 128 --save "$SAVE-128.png" --dpi 300 --ymax 11 --ymin -12
python avg_spectra.py slices-feat $NPZ_FILE $NPZ_FILE2 --key 64 --save "$SAVE-64.png" --dpi 300  --ymax 11 --ymin -4
python avg_spectra.py slices-feat $NPZ_FILE $NPZ_FILE2 --key 32 --save "$SAVE-32.png" --dpi 300 --ymax 11 --ymin -3
python avg_spectra.py slices-feat $NPZ_FILE $NPZ_FILE2 --key 16 --save "$SAVE-16.png" --dpi 300 --ymax 9 --ymin 0
python avg_spectra.py slices-feat $NPZ_FILE $NPZ_FILE2 --key 8 --save "$SAVE-8.png" --dpi 300 --ymax 7 --ymin 0

# python avg_spectra.py slices $NPZ_FILE $NPZ_FILE2 --key image --save "$SAVE-image.png" --dpi 300 --ymax 7 --ymin 0
# python avg_spectra.py slices $NPZ_FILE $NPZ_FILE2 --key 128 --save "$SAVE-128.png" --dpi 300 --ymax 7 --ymin 0
# python avg_spectra.py slices $NPZ_FILE $NPZ_FILE2 --key 64 --save "$SAVE-64.png" --dpi 300  --ymax 7 --ymin 0
# python avg_spectra.py slices $NPZ_FILE $NPZ_FILE2 --key 32 --save "$SAVE-32.png" --dpi 300 --ymax 7 --ymin 0
# python avg_spectra.py slices $NPZ_FILE $NPZ_FILE2 --key 16 --save "$SAVE-16.png" --dpi 300 --ymax 7 --ymin 0
# python avg_spectra.py slices $NPZ_FILE $NPZ_FILE2 --key 8 --save "$SAVE-8.png" --dpi 300 --ymax 7 --ymin 0