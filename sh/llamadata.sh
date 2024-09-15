python data_generation.py \
--tokenizer openlm-research/open_llama_3b_v2 \  # mamba tokenizer
--longdp Yukang/LongAlpaca-12k \   # LongAlpaca-12k path
--Orcadp Open-Orca/OpenOrca \  # OpenOrca path
--maxlen 2048 \   #max input len
--normal_datapath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcallama2048 \   # the path of noraml Orca you want to save
--long_datapath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextllama2048 \     # the path of long data you want to save