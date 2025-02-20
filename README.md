# CaloDiffusion unofficial repository (WIP) - 2.0

Implemented with Pytorch 2.0. 
Depedencies listed in the pyproject.yaml

Insall with 
```
git clone https://github.com/[fork]/CaloDiffusion.git
pip install -e .
```

# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:
* [Dataset 1](https://zenodo.org/record/6368338)
* [Dataset 2](https://zenodo.org/record/6366271)
* [Dataset 3](https://zenodo.org/record/6366324)

# Run the training scripts with

```bash
calodif-train 
    -d DATA-DIR \
    -c CONFIG \
    --checkpoint SAVE-DIR \
MODEL-TYPE
```
* Example configs in ```[config_dataset1.json/config_dataset2.json/config_dataset3.json]```'
* Additional options can be seen with `calodif-train --help`

# Sampling with the learned model

```bash
calodif-inference
  --n-events N \
  -c CONFIG \
sample MODEL-TYPE
  
```
* Additional options can be found with `calodif-inference --help`
  
# Creating the plots shown in the paper

```bash
calodif-inference
  --n-events N \
  -c CONFIG \
plot \
  --generated RESULTS-H5F
```


