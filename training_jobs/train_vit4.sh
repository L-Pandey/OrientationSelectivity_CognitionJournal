viewpoint=V11O2

for viewpoint in chick
do
    python3 ../train_vit.py \
        --max_epochs 200 \
        --batch_size 512 \
        --data_dir /data/lpandey/retinalWaves_dataset/Lalit_newSpatiotemporal_nonaugmented_dataset/newDataset_Jul2025/${viewpoint} \
        --seed_val 155 \
        --temporal \
        --window_size 3 \
        --image_size 128 \
        --patch_size 16 \
        --head 3 \
        --val_split 0.05 \
        --transforms transform_resize \
        --loss_ver v2 \
        --resize_dims 128 \
        --dataset_size 160000 \
        --exp_name /data/lpandey/LOGS/VIT_Time/KittenAI_publicationData/retinalWavesTrained/chicks_retinalWaves_controlled/ViT3H/160k/128x128_16x16_200ep_512bs/
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
