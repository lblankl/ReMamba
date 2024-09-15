
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ReMamba
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/LEval/hopeRemaba2
basepath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/LEval/hopeRemaba2
model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba2-2.7b-nohf
model_type=ReMambafast
stratio=0.0
ratio=0.5
device=cuda
compressp_ratio=0.005
peft_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/ReMamba2Thalf/63641end
tokenizer=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8btemplate
maxlen=6k
name=ReMamba2L-eval_0.005_0.5_6k
remamba_config=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba2-2.7b-nohf
echo $name
python ./Baselines/remamba.py \
--model_type $model_type \
--model_name $name \
--model_path $model_path \
--peft_path $peft_path \
--tokenizer $tokenizer \
--ratio $ratio \
--compressp_ratio $compressp_ratio \
--max_length $maxlen \
--stratio $stratio \
--metric exam_eval \
--device $device \
--base_path $basepath \
--remamba_config $remamba_config 

python Evaluation/auto_eval.py --pred_basep $basepath/Predictions/exam_eval/$name