source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ConMamba
export HF_DATASETS_CACHE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache

accelerate launch --config_file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/sh/cfg/acc.yaml /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/hopeconmambahf/Finetune.py \
--name conmaba \
--output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/conmaba  \
--dataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/concept/conceptTransformer248/datasets/OpenOrca \
--conmamba_peft /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/conmabaSim/186000 \