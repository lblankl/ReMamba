
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate Mamba2
basepath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/LEval/hopeRemaba
model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-nohf
model_type=ReMamba_branchad
stratio=0.0
ratio=0.05
device=cuda:0
compressp_ratio=0.1
peft_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/branch-adaptive/63641endpt
tokenizer=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8btemplate
maxlen=5k
name=ReMamba_branchad_debug_${compressp_ratio}_${ratio}_${maxlen}_${stratio}
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
--base_path $basepath 

python Evaluation/auto_eval.py --pred_basep $basepath/Predictions/exam_eval/$name