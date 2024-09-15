b=${basedir}
basedir=$b/Longbench
echo $basedir
# export CUDA_VISIBLE_DEVICES=0
############################################################### args to adjust
#get current path
ratio=${ratio}
compressp_ratio=${compress_ratio}
maxlen=${maxlen}
model_name=${modelname}
model_path=${model_path}
name=${name}
stratio=${stratio}
append_prompt=False
datapath=${datapath}
# ###
echo $name
peftpath=${peftpath}
##############################################################################


cd $basedir
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ReMamba


python predg.py --model $model_name --e \
--datapath $datapath \
--cfg_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/remamba \
--remamba_sample_ratio $ratio \
--out_name  $name \
--peft_path $peftpath \
--max_len $maxlen \
--append_prompt $append_prompt \
--compressp_ratio $compressp_ratio \
--stratio $stratio \
--model_path $model_path

# --task triviaqa
python eval.py --model $model_name --e  --name $name 
python avg.py --basep $basedir/pred_e/$name/
