b=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/LongBench/remamba
basedir=$b/Longbench
echo $basedir
# export CUDA_VISIBLE_DEVICES=0
############################################################### args to adjust
#get current path
ratio=0.009
compressp_ratio=0.18
maxlen=5500
model_name=ReMamba
model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf
name=ReMambalongbench_0.18_0.009_5500
stratio=0.0
append_prompt=False
# ###
echo $name
peftpath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/param/ReMamba-hf
##############################################################################


cd $basedir
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/miniconda3/bin/activate
conda activate ReMamba


python predg.py --model $model_name --e \
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
