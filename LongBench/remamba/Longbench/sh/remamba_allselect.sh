cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/LongBench
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ConMamba


# python pred.py --model remamba_allselect --e \
# --cfg_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/remamba \
# --remamba_sample_ratio 0.1 \

python eval.py --model remamba_allselect --e 