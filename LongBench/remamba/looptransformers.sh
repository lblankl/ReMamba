##########################
basedir=$(pwd)

echo $basedir

# compress_ratio_list=("0.10" "") 
max_lenlist=("1500" )
datapath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/LongBench/THUDM/LongBench
model_name=transformers
nameprefix=llamalongbench
peftpath=None
model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/openlm-research/open_llama_3b_v2
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ReMamba
#################

for maxlen in "${max_lenlist[@]}"; 
do 
   

            name=${nameprefix}_${maxlen}
            echo $name 

            python ./template.py \
            --datapath $datapath \
            --maxlen $maxlen --basedir $basedir \
            --peftpath $peftpath \
            --jobname $name  \
            --model_name $model_name \
            --model_path $model_path \

            # sh ./hopecreate.sh
            # hope run $name.hope
            sh run.sh

    
done