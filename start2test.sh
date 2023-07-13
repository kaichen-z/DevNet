num='0'
CUDA_VISIBLE_DEVICES=0 python  -u -m test --load_weights_folder logs/weights --cutmix False --use_freeze_epoch 20 --seed 1024 --scheduler_step_size 14  --batch 1  --model_name model$num --png
