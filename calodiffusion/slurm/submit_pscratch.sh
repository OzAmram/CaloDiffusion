#!/bin/bash
#SBATCH --job-name=calodif-XYZ
#SBATCH --nodes=1
#SBATCH --account=m2612
#SBATCH --qos regular
#SBATCH --constraint=gpu
#SBATCH --ntasks=1
#SBATCH -G 4
#SBATCH --time=06:00:00
#SBATCH --module=cvmfs
#SBATCH --open-mode=append     # Append output to log files
#SBATCH --requeue

# Number of times to requeue
export max_restarts=XYZ

# Any specific branch to run code off
export BRANCH='XYZ'
export REPO='https://github.com/XYZ/CaloDiffusion'

# Set up environment and DMTCP coordinator
export DMTCP_COORD_HOST=$(hostname)

# Setting data dirs
export DATA_DIR=/global/cfs/cdirs/m2612/calodiffusion
export base_dir=$DATA_DIR/HGCal_showers_william_v2
export TRAIN_DATA=$PSCRATCH/HGCal_showers_william_v2
mkdir -p $TRAIN_DATA

# Set the checkpoints and config
export CHECKPOINT_DIR=$HOME/CaloDiffusion/checkpoints
mkdir -p $CHECKPOINT_DIR
export CONFIG=$HOME/CaloDiffusion/XYZ

# Actual running command
export CALODIF_COMMAND="timeout 5.9h python3 $CLONE/calodiffusion/training.py -c $CONFIG -d $TRAIN_DATA --checkpoint $CHECKPOINT_DIR XYZ"


# Check the job restarts to decide if the job has resources
# Checking the dir isn't reliable, we can have jobs with the same name but different ID
export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
if [ "$restarts" -eq 0]; then
    # If this is the first requeue

    export id=$(uuidgen)
    export CLONE="${PSCRATCH}/calodif_clone_${SLURM_JOB_NAME}_${id}"
    mkdir $CLONE
    cd $CLONE
    git clone -b $BRANCH $REPO $CLONE
    # get the calochallenge and hgcal repos
    git clone https://github.com/OzAmram/CaloChallenge.git
    git clone https://github.com/OzAmram/HGCalShowers.git

    # Make an env
    module load python
    conda create --prefix ${CLONE}_env --clone calodif

else
    # Set the clone branch and the ID
    # Use find to list pscratch with calodif_clone_{job} and exclude the env
    export CLONE=$(find $PSCRATCH -type d -name "calodif_clone_${SLURM_JOB_NAME}_*" | grep -v "_env" |  head -n 1)
    module load python

fi

function requeue () {
    export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
    
    if [ "$restarts" -ge "$max_restarts" ]; then
        echo "Max restarts reached - restarts at $max_restarts. Not requeuing."
        
        # Remove all the stuff from the end
        conda deactivate
        conda remove ${CLONE}_env --all
        rm -rf $CLONE

        exit 0
    else
        echo "Going to requeue - at $restarts restarts"
        scontrol requeue ${SLURM_JOB_ID}
    fi
}


# Get a list of files in the source directory
files=$(ls "$base_dir")
# Loop through the list of files
for file in $files
do
  # Check if the file exists in the destination directory and only copy it if it does not exist already
  if [ ! -f "$TRAIN_DATA/$file" ]; then
    cp "$base_dir/$file" "$TRAIN_DATA/$file"
    echo "Copied $file to $TRAIN_DATA"
  fi
done

# Make a copy of the base env and replace the instance of calodif
conda activate ${CLONE}_env
pip install -e $CLONE --no-deps

# Run the training or inference
$CALODIF_COMMAND

# requeue and/or cleanup
requeue