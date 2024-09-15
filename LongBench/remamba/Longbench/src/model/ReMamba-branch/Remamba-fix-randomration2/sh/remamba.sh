source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ConMamba
export HF_DATASETS_CACHE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache

accelerate launch --config_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/hopeRemambahf/sh/default_config.yaml  ./Finetune.py \
--name ReMamba-branch-fix-randomratio2 \
--logging_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/log \
--output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/ReMamba-branch-fix-randomratio2  \
--dataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcaRemambag \
--longdataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextRemambag \
--remamba_config /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/remamba \
--grad_clip 1.0 \
--num_train_epochs 1 \
--normal_data_range 300000 \
# --debug


