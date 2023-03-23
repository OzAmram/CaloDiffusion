import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Diffu', help='Diffusion model to train')
parser.add_argument('-c', '--config', default='config_dataset2.json', help='Config file with training parameters')
parser.add_argument('-n', '--name', default='test', help='job name')
parser.add_argument("--resubmit", default = False, action='store_true')
parser.add_argument("--constraint", default = "a100|v100|p100" , help='gpu resources')
parser.add_argument("--memory", default = 16000 , help='RAM')
flags = parser.parse_args()

if(flags.name[-1] =="/"): flags.name = flags.name[:-1]
print(flags.name)

base_dir = r"\/work1\/cms_mlsim\/oamram\/CaloDiffusion\/slurm\/"
if(flags.model == 'Diffu'):
    if(not os.path.exists(flags.name)): os.system("mkdir %s" % flags.name)
    cfg_loc = flags.name + "/config.json"
    script_loc = flags.name + "/diffu_train.sh"
    if(not flags.resubmit):
        os.system("cp %s %s" % (flags.config, cfg_loc))
        os.system("cp diffu_train.sh %s" % (script_loc))
        os.system("sed -i 's/JOB_NAME/%s/' %s" % (flags.name, script_loc))
        os.system("sed -i 's/JOB_OUT/%s/' %s" % (flags.name, script_loc))
        os.system("sed -i 's/CONSTRAINT/%s/' %s" % (flags.constraint, script_loc))
        os.system("sed -i 's/MEMORY/%s/' %s" % (flags.memory, script_loc))
        os.system("sed -i 's/CONFIG/%s/' %s" % (base_dir + flags.name + "\/config.json", script_loc) )
    os.system("sbatch %s" % script_loc)

