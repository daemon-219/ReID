cd /home/mmc_zhaojiacheng/project/people_image/ReID-MGN-master

export CUDA_VISIBLE_DEVICES=2

python main.py --mode train --data_path ./data --fake_ratio 0.5 --epoch 100