# CaloDiffusion unofficial repository (WIP)


Pytorch v1.9  was used to implement all models. 

# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:
* [Dataset 1](https://zenodo.org/record/6368338)
* [Dataset 2](https://zenodo.org/record/6366271)
* [Dataset 3](https://zenodo.org/record/6366324)

# Run the training scripts with

```bash
cd scripts
python train_diffu.py  --config CONFIG
```
* Example configs in ```[config_dataset1.json/config_dataset2.json/config_dataset3.json]```

# Sampling with the learned model

```bash
python plot.py  --nevts N  --sample  --config CONFIG --model MODEL
```
# Creating the plots shown in the paper

```bash
python plot.py  --config CONFIG --model MODEL
```


