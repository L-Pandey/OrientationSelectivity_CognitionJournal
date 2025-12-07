viewpoint=V11O2

for viewpoint in ep0
do
    python3 ../train_vit.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/retinalWaves_dataset/Lalit_newSpatiotemporal_nonaugmented_dataset/chick_wong98/parsed/${viewpoint} \
        --seed_val 193 \
        --temporal \
        --window_size 3 \
        --image_size 64 \
        --patch_size 8 \
        --head 6 \
        --val_split 0.05 \
        --transforms transform_none \
        --loss_ver v2 \
        --resize_dims 64 \
        --dataset_size 160000 \
        --shuffle_temporalWindows \
        --dataloader_shuffle \
        --shuffle_frames \
        --exp_name /data/lpandey/LOGS/VIT_Time/NeurIPS2025_RW_Rebuttal/temporallyScrambedlwithScheduler/vit6h/
done


# NOTES :
# --shuffle \
# --print_model \
# set window_size in the range [1,4]
# choose loss function version from v0 and v1

#  --shuffle_frames \
#  --loss_ver v0 \
#  --shuffle_temporalWindows \
#  --dataset_size 10000 \
#  --dataloader_shuffle \
