#!/bin/bash

python train.py --name celebA --dataset_mode custom --label_dir /search/odin/tk/public_data/SelfD/CelebAMask-HQ-labelids --image_dir /search/odin/tk/public_data/SelfD/CelebAMask-HQ/CelebA-HQ-img --label_nc 19 --load_size 512 --crop_size 512 --no_instance --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 16 --niter 25 --niter_decay 5