#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --output=JOB_OUT/log.txt
#SBATCH --partition=wc_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=MEMORY
#SBATCH --constraint=(CONSTRAINT)
#SBATCH  --time=08:00:00


unset LD_PRELOAD
 
max_restarts=10
 
# just gather some information about the job
scontext=$(scontrol show job $SLURM_JOB_ID)
restarts=$(echo "$scontext" | grep -o 'Restarts=.' | cut -d= -f2)
outfile=$(echo "$scontext"  | grep 'StdOut='       | cut -d= -f2)
errfile=$(echo "$scontext"  | grep 'StdErr='       | cut -d= -f2)
timelimit=$(echo "$scontext" | grep -o 'TimeLimit=.*' | awk '{print $1}' | cut -d= -f2)
 
# term handler
# the function is executed once the job gets the TERM signal
term_handler()
{
    echo "executing term_handler at $(date)"
    if [[ $restarts -lt $max_restarts ]]; then
       # copy the logfile. will be overwritten by the 2nd run
       cp -v $outfile $outfile.$restarts
       # requeue the job and put it on hold. It's not possible to change partition otherwise
       scontrol requeuehold $SLURM_JOB_ID
       # change timelimit and partition
       #scontrol update JobID=$SLURM_JOB_ID TimeLimit=$alt_timelimit Partition=$alt_partition
       scontrol update JobID=$SLURM_JOB_ID
       # release the job. It will wait in the queue for 2 minutes before the 2nd run can start
       scontrol release $SLURM_JOB_ID
    fi
}
 
# declare the function handling the TERM signal
trap 'term_handler' TERM
 
# print some job-information
cat <<EOF
SLURM_JOB_ID:         $SLURM_JOB_ID
SLURM_JOB_NAME:       $SLURM_JOB_NAME
SLURM_JOB_PARTITION:  $SLURM_JOB_PARTITION
SLURM_SUBMIT_HOST:    $SLURM_SUBMIT_HOST
TimeLimit:            $timelimit
Restarts:             $restarts
EOF
 
# the actual computation
module load apptainer
export SINGULARITY_CACHEDIR=/work1/cms_mlsim/oamram/
export HOME=/work1/cms_mlsim/oamram/CaloDiffusion/ 
torchexec() {
apptainer exec --no-home -p --nv --bind `pwd` --bind /cvmfs --bind /cvmfs/unpacked.cern.ch --bind /work1/cms_mlsim/ --bind /wclustre/cms_mlsim/ /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:pytorch-1.9.0-cuda11.1-cudnn8-runtime-singularity "$@"
}


echo "starting computation at $(date)"
cd /work1/cms_mlsim/oamram/CaloDiffusion/scripts


if [[ $restarts -eq 0 ]]; then
# Timeout command to catch / resubmit before 8h max
torchexec bash -c "export HOME=/work1/cms_mlsim/oamram/CaloDiffusion/; timeout 7.8h python3 train_MODELTYPE.py --model Diffu --config CONFIG --data_folder /wclustre/cms_mlsim/denoise/CaloChallenge/ --load --reset_training"
else 
torchexec bash -c "export HOME=/work1/cms_mlsim/oamram/CaloDiffusion/; timeout 7.8h python3 train_MODELTYPE.py --model Diffu --config CONFIG --data_folder /wclustre/cms_mlsim/denoise/CaloChallenge/ --load "
fi

#resubmit
if [[ $? -eq 124 ]]; then
    term_handler
fi

echo "all done at $(date)"
