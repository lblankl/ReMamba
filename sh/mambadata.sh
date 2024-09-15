python data_generation.py \
--tokenizer state-spaces/mamba-2.8b-hf \  # mamba tokenizer
--longdp Yukang/LongAlpaca-12k \   # LongAlpaca-12k path
--Orcadp Open-Orca/OpenOrca \  # OpenOrca path
--maxlen 6000 \   #max input len
--normal_datapath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcaRemamba6000 \   # the path of noraml Orca you want to save
--long_datapath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextRemamba6000 \     # the path of long data you want to save