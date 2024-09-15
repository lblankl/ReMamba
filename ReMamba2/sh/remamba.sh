source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ReMamba
export HF_DATASETS_CACHE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache
#--name tensorboard  # logging name  --logging_dir  # the tensorboard dir   --output_dir # the model ckpt dir  --dataset_name # normal orca data  --longdataset_name # long ctx data
#--remamba_config  # cfg path same as mamba_path --normal_data_range  # we use 300k of normal orca data  --mamba_path base mambapath
accelerate launch --config_file ./sh/acczero2.yaml  ./Finetune.py \
--name ReMamba2 \
--logging_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/ReMamba2/log \
--output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/ReMamba2/out/ReMamba2  \
--dataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcaRemambag \
--longdataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextRemambag \
--grad_clip 1.0 \
--num_train_epochs 1 \
--normal_data_range 300000 \
--gradient_accumulation_steps 16 \
--save_steps 3000 \
--mamba_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba2-2.7b-nohf \
--remamba_config /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba2-2.7b-nohf \

