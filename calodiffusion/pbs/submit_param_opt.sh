#!/bin/bash
#PBS -N hgcal_layer_opt
#PBS -A CaloDiffusion
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -l walltime=01:00:00
#PBS -q debug
# Change to the submission directory
cd $PBS_O_WORKDIR

ml use /soft/modulefiles
ml spack-pe-base/0.8.1
ml use /soft/spack/testing/0.8.1/modulefiles
ml apptainer/main
ml load e2fsprogs


# Submit a job to a PBS job queue 
# Reference: https://github.com/argonne-lcf/container-registry/blob/main/containers/datascience/pytorch/Sunspot/job_submission.sh

# set the proxies so that the job can access the internet
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128


DATA_FOLDER=/eagle/CaloDiffusion/HGCal_showers_william_v2/
CONFIG=$HOME/CaloDiffusion/configs/optimize_hgcal_sampler.json
NAME=hgcal_layer_hyperparameter_search_DEBUG
RESULT_FOLDER=$HOME/CaloDiffusion/optuna_studies/
MODEL_LOC="$HOME/CaloDiffusion/checkpoints/hgcal_Diffusion/final.pth"
LAYER_MODEL_LOC="$HOME/CaloDiffusion/checkpoints/hgcal_layer_Diffusion/final.pth"

CALODIF_COMMAND="python3 $HOME/CaloDiffusion/CaloDiffusion/calodiffusion/optimize.py --hgcal -c $CONFIG --data-folder $DATA_FOLDER --results-folder $RESULT_FOLDER -n 10 --name $NAME sample --model-loc $MODEL_LOC layer --layer-model $LAYER_MODEL_LOC"

echo "RUNNING $CALODIF_COMMAND"

############## USING APPTAINER - CANNOT USE WITH MPI ##############
# Existing container at $CONTAINER=/home/voetberg/containers/pytorch:25.04-py3.sing
# Build using apptainer build --fakeroot $CONTAINER docker://nvcr.io/nvidia/pytorch:25.04-py3

CONTAINER=/eagle/CaloDiffusion/pytorch:25.04-py3.sing
INSTALL="python3 -m pip install --no-cache-dir -e $HOME/CaloDiffusion/CaloDiffusion/ --user"

# breaks if you bind /opt - different version of torch in there and it's not compatible
apptainer exec --fakeroot -B /soft -B /eagle $CONTAINER /bin/bash -c "$INSTALL; $CALODIF_COMMAND"