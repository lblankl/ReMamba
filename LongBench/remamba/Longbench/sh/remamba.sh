# export CUDA_VISIBLE_DEVICES=0

# python pred.py --model remamba --e \
# --cfg_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/remamba \
# --remamba_sample_ratio 0.4 \

python eval.py --model remamba --e 