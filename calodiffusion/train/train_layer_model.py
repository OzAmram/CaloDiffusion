from calodiffusion.models.layerdiffusion import LayerDiffusion
from calodiffusion.train.train_diffusion import TrainDiffusion

class TrainLayerModel(TrainDiffusion):
    def __init__(self, flags, config, load_data = True, inference=False, save_model=True, *args, **kwargs):
        super().__init__(flags, config, load_data, save_model=save_model, *args, **kwargs)
        self.init_model()
        if inference: 
            self.model.set_layer_state(False)
        else: 
            self.model.set_layer_state(True)

    def init_model(self):
        self.model = LayerDiffusion(
            self.config, n_steps = self.config["NSTEPS"], loss_type = self.config['LOSS_TYPE']
        )