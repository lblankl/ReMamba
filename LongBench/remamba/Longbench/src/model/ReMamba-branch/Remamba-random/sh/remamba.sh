source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ConMamba
export HF_DATASETS_CACHE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache

accelerate launch --config_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/hopeRemambahf/sh/default_config.yaml  ./Finetune.py \
--name Hybrid-nogist \
--logging_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/log \
--output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/Hybrid-nogist  \
--dataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcaRemambag \
--longdataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextRemambag \
--remamba_config /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/remamba \
--grad_clip 1.0 \
--num_train_epochs 1 \
--normal_data_range 300000 \
--select_rate_Max 0.1 \
--select_rate_Min 0.1 \
# --data_point 240000 \
# --remamba_peft /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/Remamba-gist-Hybrid/30000 \

# --debug


