#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=JOB_OUT/log_plot_JOBIDX.txt
#SBATCH --partition=wc_gpu
#SBATCH --nodes=1
#DOGPU --gres=TYPE:1
#DOGPU --constraint=(CONSTRAINT)
#SBATCH --mem=MEMORY
set -x


module load apptainer
export SINGULARITY_CACHEDIR=/work1/cms_mlsim/oamram/
export HOME=/work1/cms_mlsim/oamram/CaloDiffusion/ 
torchexec() {
apptainer exec --no-home -p --nv --bind `pwd` --bind /cvmfs --bind /cvmfs/unpacked.cern.ch --bind /work1/cms_mlsim/ --bind /wclustre/cms_mlsim/ /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:pytorch-1.9.0-cuda11.1-cudnn8-runtime-singularity "$@"
}

cd /work1/cms_mlsim/oamram/CaloDiffusion/scripts
torchexec bash -c "export HOME=/work1/cms_mlsim/oamram/CaloDiffusion/; python plot.py --config MDIR/config.json --model MODEL --data_folder /wclustre/cms_mlsim/denoise/CaloChallenge/ --model_loc MDIR/MNAME --plot_folder ../plots/JOB_NAME --nevts NEVTS  --batch_size BATCH_SIZE --sample --sample_algo SAMPLE_ALGO --sample_offset SAMPLE_OFFSET --sample_steps SAMPLE_STEPS --job_idx JOBIDX --layer_model LAYMODEL"

EVAL=EVAL_VAR

if [ "$EVAL" = true ]; then

    cd /work1/cms_mlsim/oamram/CaloDiffusion/CaloChallenge/code
    torchexec python evaluate.py -i MDIR/generated_MTAG.h5 -r /wclustre/cms_mlsim/denoise/CaloChallenge/dataset_2_eval_25k.h5 -d 2 --output_dir ../../plots/JOB_NAME/challenge_plots/ -m all
fi
exit
