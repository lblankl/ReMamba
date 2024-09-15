##########################
basedir=$(pwd)

echo $basedir

compress_ratio_list=( "0.18") 
# compress_ratio_list=("0.10" "") 
max_lenlist=("5500" )
datapath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/LongBench/THUDM/LongBench
#("1500" "2500" "4500" "5500")
ratio_list=("0.009" ) 
model_name=mamba2fast
nameprefix=mamba2fastlongbench
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
            name=${nameprefix}_${maxlen}
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