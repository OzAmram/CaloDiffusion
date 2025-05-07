#!/bin/bash
#PBS -N debug_sub
#PBS -A CaloDiffusion
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -l walltime=00:30:00
#PBS -q debug

ml use /soft/modulefiles
ml spack-pe-base/0.8.1
ml use /soft/spack/testing/0.8.1/modulefiles
ml apptainer/main
ml load e2fsprogs

export BASE_SCRATCH_DIR=/local/scratch/ # For Polaris
export APPTAINER_TMPDIR=$BASE_SCRATCH_DIR/apptainer-tmpdir
mkdir -p $APPTAINER_TMPDIR

export APPTAINER_CACHEDIR=$BASE_SCRATCH_DIR/apptainer-cachedir
mkdir -p $APPTAINER_CACHEDIR


export CONTAINER="${APPTAINER_CACHEDIR}/pytorch:25.04-py3.sing"

# Submit a job to a PBS job queue 
# Reference: https://github.com/argonne-lcf/container-registry/blob/main/containers/datascience/pytorch/Sunspot/job_submission.sh

# set the proxies so that the job can access the internet
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Change to the submission directory
cd $PBS_O_WORKDIR

# Build the sing 
# 25.04 contains pytorch 2.7a
apptainer build --fakeroot $CONTAINER docker://nvcr.io/nvidia/pytorch:25.04-py3

export DATA_FOLDER=/eagle/CaloDiffusion
export CONFIG=./CaloDiffusion/test_config.json

export CALODIF_COMMAND=calodif-train -c $CONFIG -d $DATA_FOLDER -n 10 diffusion

# Command to run a command in the container
apptainer exec --nv --fakeroot --bind $SCRATCH:/$BASE_SCRATCH_DIR --bind $HOME:/home --bind /eagle:/eagle $CONTAINER /bin/bash -c "export PATH=/root/.local/bin:$PATH; python3 -m pip install --no-cache-dir -e ./CaloDiffusion --user; $CALODIF_COMMAND"