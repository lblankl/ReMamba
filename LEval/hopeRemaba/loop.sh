##########################
basedir=$(pwd)

echo $basedir

compress_ratio_list=( "0.20" ) 
# compress_ratio_list=("0.10" "") 
max_lenlist=("5k" "6k" "7k" "8k" )
#("1500" "2500" "4500" "5500")
ratio_list=("0.05" ) 
model_name=ReMamba
nameprefix=ReMambaL-eval
model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf
peftpath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/branch-adaptive-Ti-od/63641end
datapath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/SuperGist/InfiniteBench/data
tokenizer=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8btemplate
remamba_config=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/remamba
device=cuda
env=ReMamba
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate

conda activate Mamba2
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
            --ratio $ratio \
            --compress_ratio $compress_ratio \
            --maxlen $maxlen --basedir $basedir \
            --peftpath $peftpath \
            --jobname $name  \
            --model_name $model_name \
            --datapath $datapath \
            --tokenizer $tokenizer \
            --remamba_config $remamba_config \
            --device $device \
            --env $env \
            --model_path $model_path
            sh ./hopecreate.sh
            hope run $name.hope

    
        done
    done
done