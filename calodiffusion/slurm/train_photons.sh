#!/bin/bash
#SBATCH --job-name=photon-train-calodif
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

# Set up environment and DMTCP coordinator
export DMTCP_COORD_HOST=$(hostname)
export DATA_DIR=/global/cfs/cdirs/m2612/calodiffusion

#export base_dir=$DATA_DIR/HGCal_sim_samples/SinglePion/
#export TRAIN_DATA=$PSCRATCH/HGCal_sim_samples/SinglePion/
#export CONFIG=$HOME/CaloDiffusion/calodiffusion/configs/config_HGCal_pions.json

export base_dir=$DATA_DIR/HGCal_sim_samples/SinglePhoton/
export TRAIN_DATA=$PSCRATCH/HGCal_sim_samples/SinglePhoton/
export CONFIG=$HOME/CaloDiffusion/calodiffusion/configs/config_HGCal_photons.json
mkdir -p $TRAIN_DATA

# Watch for the job ending to resubmit
export max_restarts=9
function requeue () {
    export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
    if [ "$restarts" -ge "$max_restarts" ]; then
        echo "Max restarts reached - restarts at $max_restarts. Not requeuing."
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


export CHECKPOINT_DIR=$HOME/CaloDiffusion/checkpoints
mkdir -p $CHECKPOINT_DIR

export CALODIF_COMMAND="timeout 5.9h python3 $HOME/CaloDiffusion/calodiffusion/training.py -c $CONFIG -d $TRAIN_DATA --checkpoint $CHECKPOINT_DIR --hgcal diffusion"
export CALODIF_LOAD_COMMAND="timeout 5.9h python3 $HOME/CaloDiffusion/calodiffusion/training.py -c $CONFIG -d $TRAIN_DATA --checkpoint $CHECKPOINT_DIR --hgcal --load diffusion"


module load python
conda activate calodif

cd $HOME/CaloDiffusion

export restarts=$(scontrol show jobid $SLURM_JOB_ID | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)
echo "Restart $restarts"

if [[ $restarts -eq 0 ]]; then
    echo $CALODIF_LOAD_COMMAND
    $CALODIF_LOAD_COMMAND
else
    echo $CALODIF_LOAD_COMMAND
    $CALODIF_LOAD_COMMAND
fi

requeue
