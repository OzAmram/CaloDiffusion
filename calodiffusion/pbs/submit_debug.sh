#!/bin/bash
#PBS -N debug_sub
#PBS -A CaloDiffusion
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:30:00
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


DATA_FOLDER=/eagle/CaloDiffusion
# Replace with given config
CONFIG=$HOME/CaloDiffusion/CaloDiffusion/test_config.json 
CHECKPOINT_FOLDER=$HOME/CaloDiffusion/checkpoints
# n = 10 for debug
CALODIF_COMMAND="python3 $HOME/CaloDiffusion/CaloDiffusion/calodiffusion/training.py -c $CONFIG -d $DATA_FOLDER -n 10 --checkpoint $CHECKPOINT_FOLDER diffusion"

echo "RUNNING $CALODIF_COMMAND"

############## USING APPTAINER - CANNOT USE WITH MPI ##############
# Existing container at $CONTAINER=/home/voetberg/containers/pytorch:25.04-py3.sing
# Build using apptainer build --fakeroot $CONTAINER docker://nvcr.io/nvidia/pytorch:25.04-py3

CONTAINER=$HOME/containers/pytorch:25.04-py3.sing
INSTALL="python3 -m pip install --no-cache-dir -e $HOME/CaloDiffusion/CaloDiffusion/ --user"

# breaks if you bind /opt - different version of torch in there and it's not compatible
apptainer exec --fakeroot -B /soft -B /eagle $CONTAINER /bin/bash -c "$INSTALL; $CALODIF_COMMAND"


############## USING MPI ##############


# module load conda
# module load cray-hdf5-parallel/1.12.2.9
# conda activate base

# CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
# VENV_DIR="$(pwd)/venvs/${CONDA_NAME}"
# # Only has to be done once
# # mkdir -p "${VENV_DIR}"
# # python -m venv "${VENV_DIR}" --system-site-packages

# source "${VENV_DIR}/bin/activate"

# # Instructions recommend to use `ignore-installed` but this doesn't work with our version of pytorch
# python3 -m pip install --no-cache-dir -e $HOME/CaloDiffusion/CaloDiffusion/

# # Environment variables for MPI
# export ADDITIONAL_PATH=/opt/cray/pe/pals/1.2.12/lib
# module load cray-mpich-abi
# export APPTAINERENV_LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:$ADDITIONAL_PATH"

# # Set MPI ranks
# NODES=$(wc -l < $PBS_NODEFILE)
# PPN=16
# PROCS=$((NODES * PPN))
# echo "NUM_OF_NODES=${NODES}, TOTAL_NUM_RANKS=${PROCS}, RANKS_PER_NODE=${PPN}"

# mpiexec -hostfile $PBS_NODEFILE -n $PROCS -ppn $PPN $CALODIF_COMMAND