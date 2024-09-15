##########################
basedir=$(pwd)

echo $basedir
# ratio_list corresponding to ρ list. compress_ratio_list  corresponding to p list. s is 0 here.  max_lenlist is the max ctx length we just use the default setting in long bench: 2k:1500 3k:2500 4k:3500
compress_ratio_list=( "0.25") 

max_lenlist=("5500" )

ratio_list=("0.05" ) 
# model_name is for recognizing model type can not be changed
model_name=ReMamba2fast
# nameprefix can be change to anything  just for distinguishing
nameprefix=ReMamba2longbench
#datapath is the path for longbench data
datapath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/LongBench/THUDM/LongBench
# peftpath is the lora path saved during training
peftpath=None
model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba2-2.7b-nohf
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ReMamba
#################

for maxlen in "${max_lenlist[@]}"; 
do 
   
    for compress_ratio in "${compress_ratio_list[@]}";
    do 
        for ratio in "${ratio_list[@]}";
        do
            name=${nameprefix}_${compress_ratio}_${ratio}_${maxlen}
            echo $name 

            python ./template.py \
            --datapath $datapath \
            --ratio $ratio \
            --compress_ratio $compress_ratio \
            --maxlen $maxlen --basedir $basedir \
            --peftpath $peftpath \
            --jobname $name  \
            --model_name $model_name \
            --model_path $model_path \

            # sh ./hopecreate.sh
            # hope run $name.hope
            sh run.sh

    
        done
    done
done