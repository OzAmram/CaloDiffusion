from calodiffusion.models.layerdiffusion import LayerDiffusion
from calodiffusion.train.train_diffusion import Diffusion

class TrainLayerModel(Diffusion):
    def __init__(self, flags, config, load_data = True, inference=False):
        super().__init__(flags, config, load_data)
        if inference: 
            self.model.layer_loss = False

    def init_model(self):
        self.model = LayerDiffusion(
            self.config, n_steps = self.config["NSTEPS"], loss_type = self.config['LOSS_TYPE']
        )