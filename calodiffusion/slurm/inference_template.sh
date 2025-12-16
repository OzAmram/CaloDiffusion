#!/bin/bash
#SBATCH --job-name=JOBNUM-inf-photon
#SBATCH --nodes=1
#SBATCH --account=m2612
#SBATCH --qos regular
#SBATCH --constraint=gpu
#SBATCH --ntasks=1
#SBATCH -G 4
#SBATCH --time=03:00:00
#SBATCH --module=cvmfs
#SBATCH --open-mode=append     # Append output to log files
#SBATCH --requeue

# Set up environment and DMTCP coordinator
export DMTCP_COORD_HOST=$(hostname)
export DATA_DIR=/global/cfs/cdirs/m2612/calodiffusion

#export base_dir=$DATA_DIR/HGCal_sim_samples/SinglePhoton/
#export TRAIN_DATA=$PSCRATCH/HGCal_sim_samples/SinglePhoton/
#export CONFIG=$HOME/CaloDiffusion/calodiffusion/configs/config_HGCal_photons.json
#export MODEL=HGCal_photon_april14
#export BATCH_SIZE=100

export base_dir=$DATA_DIR/HGCal_sim_samples/SinglePion/
export TRAIN_DATA=$PSCRATCH/HGCal_sim_samples/SinglePion/
export CONFIG=$HOME/CaloDiffusion/calodiffusion/configs/config_HGCal_pions.json
export MODEL=HGCal_pion_oct17
export BATCH_SIZE=10

export CHECKPOINT_DIR=$HOME/CaloDiffusion/checkpoints
export ODIR=OUTDIR
mkdir -p $ODIR


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


export CALODIF_COMMAND="timeout 5.9h python3 $HOME/CaloDiffusion/calodiffusion/training.py -c $CONFIG -d $TRAIN_DATA --checkpoint $CHECKPOINT_DIR --hgcal diffusion"
export INF_COMMAND="python calodiffusion/inference.py  -c $CONFIG -d $TRAIN_DATA  sample --model-loc $CHECKPOINT_DIR/${MODEL}_Diffusion/checkpoint.pth --sample-algo DDim --sample-steps 200 --sample-file SAMPLE_FILE --sparse-decoding --batch-size $BATCH_SIZE -g $ODIR/batch_JOBNUM.h5 layer --layer-model $CHECKPOINT_DIR/${MODEL}_LayerModel/checkpoint.pth" 


module load python
conda activate calodif

cd $HOME/CaloDiffusion
echo $INF_COMMAND
$INF_COMMAND
