
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ${env}
cd ${basedir}
basepath=${basedir}
model_path=${model_path}
model_type=${modelname}
stratio=${stratio}
ratio=${ratio}
device=${device}
compressp_ratio=${compress_ratio}
peft_path=${peftpath}
tokenizer=${tokenizer}
maxlen=${maxlen}
name=${name}
remamba_config=${remamba_config}
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