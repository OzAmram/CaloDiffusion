# Caloscore official repository

Repo for diffusion based calorimeter generation.

Started as an offshoot of
[CaloScore](https://github.com/ViniciusMikuni/CaloScore).

Following the approach of [Denoising Diffusion Probabilistic
Models](https://arxiv.org/abs/2006.11239) and [High-Resolution Image Synthesis
with Latent Diffusion Models](http://arxiv.org/abs/2112.10752).



[Tensorflow 2.6.0](https://www.tensorflow.org/) was used to implement all models.

# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:
* [Dataset 1](https://zenodo.org/record/6368338)
* [Dataset 2](https://zenodo.org/record/6366271)
* [Dataset 3](https://zenodo.org/record/6366324)

# Run the training scripts with

```bash
cd scripts
python train.py  --config CONFIG --model MODEL
```
* MODEL options are: subVPSDE/VESDE/VPSDE (Original CaloScore)
* For training an autoencoder or training a diffusion model, use `train_ae.py` or `train_diffu.py`

* CONFIG options are ```[config_dataset1.json/config_dataset2.json/config_dataset3.json]```

# Sampling from the learned score function

```bash
python plot.py  --nevts N  --sample  --config CONFIG --model MODEL
```
# Creating the plots shown in the paper

```bash
python plot.py  --config CONFIG --model MODEL
```


