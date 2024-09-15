import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default='dd')
parser.add_argument("--ratio", type=str, default='0.1')
#compressp_ratio
parser.add_argument("--compress_ratio", type=str, default='0.1')
#maxlen
parser.add_argument("--maxlen", type=int, default=1500)
#basedir
parser.add_argument("--basedir", type=str, default="data")
#peftpath
parser.add_argument("--peftpath", type=str, default="peft")
#job name
parser.add_argument("--jobname", type=str, default="peft")
#stratio
parser.add_argument("--stratio", type=str, default="0.0")
#modelname
parser.add_argument("--model_name", type=str, default="model")
#model_path
parser.add_argument("--model_path", type=str, default="model")

with open("template.sh") as f:
        template = f.read()
args = parser.parse_args()
template = template.replace("${ratio}", str(args.ratio))
template = template.replace("${compress_ratio}", str(args.compress_ratio))
template = template.replace("${maxlen}", str(args.maxlen))
template = template.replace("${basedir}", args.basedir)
template = template.replace("${peftpath}", args.peftpath)
template = template.replace("${stratio}", args.stratio)
template = template.replace("${name}", args.jobname)
template = template.replace("${modelname}", args.model_name)
template = template.replace("${model_path}", args.model_path)
template = template.replace("${datapath}", args.datapath)

with open('run.sh', "w") as f:
        f.write(template)


# with open("creatjob.sh") as f:
#         template = f.read()
# template = template.replace("${jobname}", args.jobname)
# with open('hopecreate.sh', "w") as f:
#         f.write(template)