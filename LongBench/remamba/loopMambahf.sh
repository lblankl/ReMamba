##########################
basedir=$(pwd)

echo $basedir

datapath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/LongBench/THUDM/LongBench
# compress_ratio_list=("0.10" "") 
max_lenlist=("5500" )
#("1500" "2500" "4500" "5500")
model_name=mamba
nameprefix=Mambahflongbench
peftpath=None
model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf
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