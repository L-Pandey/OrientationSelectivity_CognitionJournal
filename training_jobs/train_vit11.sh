viewpoint=V11O2

for viewpoint in ep0
do
    python3 ../train_vit.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/retinalWaves_dataset/Lalit_newSpatiotemporal_nonaugmented_dataset/chick_wong98/parsed/${viewpoint} \
        --seed_val 50 \
        --temporal \
        --window_size 3 \
        --image_size 64 \
        --patch_size 8 \
        --head 2 \
        --val_split 0.05 \
        --transforms transform_none \
        --loss_ver v2 \
        --resize_dims 64 \
        --dataset_size 80000 \
        --exp_name /data/lpandey/LOGS/VIT_Time/KittenAI_publicationData/retinalWavesTrained/chick_retinalWaves/ViT2H/80k/
done

for viewpoint in ep0
do
    python3 ../train_vit.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/retinalWaves_dataset/Lalit_newSpatiotemporal_nonaugmented_dataset/chick_wong98/parsed/${viewpoint} \
        --seed_val 90 \
        --temporal \
        --window_size 3 \
        --image_size 64 \
        --patch_size 8 \
        --head 2 \
        --val_split 0.05 \
        --transforms transform_none \
        --loss_ver v2 \
        --resize_dims 64 \
        --dataset_size 80000 \
        --exp_name /data/lpandey/LOGS/VIT_Time/KittenAI_publicationData/retinalWavesTrained/chick_retinalWaves/ViT2H/80k/
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