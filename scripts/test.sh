cd /home/mmc_zhaojiacheng/project/people_image/ReID-MGN-master

export CUDA_VISIBLE_DEVICES=3

python main.py --mode evaluate --data_path ./data --weight ./checkpoints/best.pt