
# cd Remamba-hf
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ReMamba
export HF_DATASETS_CACHE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache

#--name tensorboard  # logging name  --logging_dir  # the tensorboard dir   --output_dir # the model ckpt dir  --dataset_name # normal orca data  --longdataset_name # long ctx data
#--remamba_config  # cfg path  --normal_data_range  # we use 300k of normal orca data   --mamba_path base mambapath
accelerate launch --config_file ./sh/acc.yaml  ./Finetune.py \
--name ReMambahf \
--logging_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/Remambahf/log \
--output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/Remambahf/out/ReMambahf \
--dataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcaRemambag \
--longdataset_name /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextRemambag \
--remamba_config ./remambacfg \
--grad_clip 1.0 \
--num_train_epochs 1 \
--normal_data_range 300000 \
--select_rate_Max 0.1 \
--select_rate_Min 0.1 \
--gradient_accumulation_steps 16 \
--save_steps 3000 \
--mamba_path state-spaces/mamba-2.8b \
--remamba_config state-spaces/mamba2-2.8b \
