import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Diffu', help='Model for plotting')
parser.add_argument('-d', '--model_dir', default='../models/TEST', help='Directory containing saved model')
parser.add_argument('-n', '--name', default='test', help='job name')
parser.add_argument('-v', '--model_version', default='checkpoint.pth', help='Which model to plot (best_val.pth, checkpoint.pth, final.pth)')
parser.add_argument('--sample_algo', default='euler', help='Sampling algo')
parser.add_argument('--sample_offset', default=0, type = int, help='Offset for sampling')
parser.add_argument('--nevts', default=1000, type = int, help='Offset for sampling')
parser.add_argument('--eval', default=False, action = 'store_true', help='Run CaloChallenge eval metrics')
parser.add_argument("--constraint", default = "a100|v100|p100" , help='gpu resources')
parser.add_argument("--memory", default = 16000 , help='RAM')
flags = parser.parse_args()

base_dir = r"\/work1\/cms_mlsim\/oamram\/CaloDiffusion\/models\/"
if(flags.model_dir[-1] != "/"): flags.model_dir += "/"
if(flags.name[-1] == "/"): flags.name = flags.name[:-1]
model_dir_tail = flags.model_dir.split("/")[-2]
if(flags.model == 'Diffu'):
    if(not os.path.exists(flags.name)): os.system("mkdir %s" % flags.name)
    script_loc = flags.name + "/plot.sh"
    os.system("cp plot.sh %s" % (script_loc))
    os.system("sed -i 's/JOB_NAME/%s/g' %s" % (flags.name, script_loc))
    os.system("sed -i 's/JOB_OUT/%s/g' %s" % (flags.name, script_loc))
    os.system("sed -i 's/MODEL/%s/g' %s" % (flags.model, script_loc) )
    os.system("sed -i 's/MDIR/%s/g' %s" % (base_dir +  model_dir_tail, script_loc) )
    os.system("sed -i 's/MNAME/%s/g' %s" % (flags.model_version, script_loc) )
    os.system("sed -i 's/SAMPLE_ALGO/%s/g' %s" % (flags.sample_algo, script_loc) )
    os.system("sed -i 's/SAMPLE_OFFSET/%s/g' %s" % (flags.sample_offset, script_loc) )
    os.system("sed -i 's/NEVTS/%s/g' %s" % (str(flags.nevts), script_loc) )

    os.system("sed -i 's/EVAL_VAR/%s/g' %s" % (str(flags.eval).lower(), script_loc) )
    os.system("sed -i 's/MTAG/%s/g' %s" % (model_dir_tail, script_loc) )

    os.system("sed -i 's/CONSTRAINT/%s/' %s" % (flags.constraint, script_loc))
    os.system("sed -i 's/MEMORY/%s/' %s" % (flags.memory, script_loc))


    os.system("sbatch %s" % script_loc)

