cd /home/mmc_zhaojiacheng/project/people_image/ReID-MGN-master

export CUDA_VISIBLE_DEVICES=3

python main.py --mode plot --data_path ./data --weight ./weights/model_100.0.pt