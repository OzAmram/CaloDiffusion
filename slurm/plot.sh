#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=JOB_OUT/log_plot.txt
#SBATCH --partition=gpu_gce
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=(a100|p100|v100)
#BATCH  --time=23:45:00
set -x


module load singularity
export SINGULARITY_CACHEDIR=/work1/cms_mlsim/
export HOME=/work1/cms_mlsim/CaloDiffusion/ 
torchexec() {
singularity exec --no-home -p --nv --bind `pwd` --bind /cvmfs --bind /cvmfs/unpacked.cern.ch --bind /work1/cms_mlsim/ --bind /wclustre/cms_mlsim/ /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:pytorch-1.9.0-cuda11.1-cudnn8-runtime-singularity "$@"
}

cd /work1/cms_mlsim/CaloDiffusion/scripts
export HOME=/work1/cms_mlsim/CaloDiffusion/ 
torchexec bash -c "export HOME=/work1/cms_mlsim/CaloDiffusion/; python plot.py --config MDIR/config.json --model MODEL --data_folder /wclustre/cms_mlsim/denoise/CaloChallenge/ --model_loc MDIR/MNAME --plot_folder ../plots/JOB_NAME --nevts NEVTS  --batch_size 100 --sample --sample_algo SAMPLE_ALGO --sample_offset SAMPLE_OFFSET"
exit
