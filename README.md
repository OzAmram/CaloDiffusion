# CaloDiffusion unofficial repository (WIP) - 2.0

Implemented with Pytorch 2.0. 
Dependencies listed in the pyproject.yaml

Install with 
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
* Example configs in ```[config_dataset1.json/config_dataset2.json/config_dataset3.json]```
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

# Repository Structure and Contributing

This repository is broken into 4 main parts:

## 1. Scripts 

`calodiffusion/inference.py` and `calodiffusion/train.py` allow for CLI based inference and training, consider them the client for the rest of the repository. 
Functionality can be seen using `--help` menus. 

## 2. Train 

The base `calodiffusion/train/train.py` class is an abstract class that can load all the necessary functions and data for training a model. 
It also contains saving methods. 
It is a "driver" class, only providing minimal instructions on how to iterate through batches of data during training or inference. 
Subclasses of `Train` have two necessary functions - `init_model` and `training_loop`. 
Init model returns a specific initialized `calodiffusion/model/diffusion` object that will be used in training, and `training_loop` defines how a single batch of data is processed.
`calodiffusion/train/evaluation.py` contains extra metrics to quantify the success of training. 

## 3. Models

### Diffusion Models 
The base `model` class in `calodiffusion` is `calodiffusion/models/diffusion.py`. 
This abstract class contains methods for loading a specific sampler for inference, which loss is used, and how training or inference forward passes are performed. 
It can also define specific ways `.pt` trained weights are loaded in the case of models with different moving pieces. 
`Diffusion` is meant to have a (or several) `pytorch.nn.module` attributes to be used, not be a subclass of `pytorch.nn.module` itself. 

A subclass of `Diffusion` has 4 required functions: 

* init_model - provide a pytorch.nn.module object to be assigned to self.model
* forward - define how that model takes data and provides a prediction. Can be as simple as calling model.forward(). Called during training.
* \_\_call__ - Define how denoising is done for a specific model. Called during inference.
* noise_generation - Generate noise for each inference step. Provides a generic "default_noise" option, but each subclass must confirm that they are using this default. 

### Samplers 
Functions used in the denoising process to condition input for each step in the process.
Additional settings for each sampler can be set using the `SAMPLER_OPTIONS` of the configuration. 
The selected sampler is using in `diffusion.sample`.  


### Loss

Loss is calculated in 2 stages - the loss metric, and then the loss calculation. 
Metrics can be mixed and matched with calculations. 
These are both set in the config.json file.
The metric defines how the prediction is processed to be compared with ground truth values, and the calculation defines how they are numerically compared (using an L1 loss, MSE, etc). 

## 4. Utils
A catch-all category for small utility functions used across training, inference, and evaluation. 