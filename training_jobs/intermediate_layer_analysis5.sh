
# untrained ViT 5 H

python3 /home/lpandey/Working_Directory/paperRW/ChicksAndDNNs_ViewInvariance/evaluate_intermediateLayers.py --data_dir "/data/lpandey/KittenAI_Dataset/LinearProbeTrainTest/" --project_name "KittenAI_publicationData" --exp_name "Untrained_ViT5H_seed0" --model "untrained_vit" --total_model_layers 5 --max_epochs 100 --shuffle "True" --img_res 64

python3 /home/lpandey/Working_Directory/paperRW/ChicksAndDNNs_ViewInvariance/evaluate_intermediateLayers.py --data_dir "/data/lpandey/KittenAI_Dataset/LinearProbeTrainTest/" --project_name "KittenAI_publicationData" --exp_name "Untrained_ViT5H_seed1" --model "untrained_vit" --total_model_layers 5 --max_epochs 100 --shuffle "True" --img_res 64

python3 /home/lpandey/Working_Directory/paperRW/ChicksAndDNNs_ViewInvariance/evaluate_intermediateLayers.py --data_dir "/data/lpandey/KittenAI_Dataset/LinearProbeTrainTest/" --project_name "KittenAI_publicationData" --exp_name "Untrained_ViT5H_seed2" --model "untrained_vit" --total_model_layers 5 --max_epochs 100 --shuffle "True" --img_res 64
