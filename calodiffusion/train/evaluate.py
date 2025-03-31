"""

General evaluation metrics for a fully trained model (not losses)

"""
from typing import Literal
from calodiffusion.utils import utils
import numpy as np
import torch 

from torchvision.models.resnet import ResNet, BasicBlock

import calodiffusion.utils.HighLevelFeatures as HLF

try: 
    import jetnet
except ImportError:
    jetnet = None

class FDPCalculationError(Exception): 
    def __init__(self, *args):
        super().__init__(*args)

class FDP: 
    def __init__(self, binning_dataset, particle): 
        if jetnet is None: 
            raise ImportError("jetnet is not installed. Please install it to use FDP evaluation.")
        
        self.hlf = HLF.HighLevelFeatures(particle, filename=binning_dataset)
        self.reference_hlf = HLF.HighLevelFeatures(particle, filename=binning_dataset)

    def pre_process(self, energies, hlf_class, label): 
        """ takes hdf5_file, extracts high-level features, appends label, returns array """
        E_layer = []
        for layer_id in hlf_class.GetElayers():
            E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
        EC_etas = []
        EC_phis = []
        Width_etas = []
        Width_phis = []
        for layer_id in hlf_class.layersBinnedInAlpha:
            EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
            EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
            Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
            Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
        E_layer = np.concatenate(E_layer, axis=1)
        EC_etas = np.concatenate(EC_etas, axis=1)
        EC_phis = np.concatenate(EC_phis, axis=1)
        Width_etas = np.concatenate(Width_etas, axis=1)
        Width_phis = np.concatenate(Width_phis, axis=1)
        ret = np.concatenate([np.log10(energies), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                            Width_etas/1e2, Width_phis/1e2, label*np.ones_like(energies)], axis=1)
        return ret

    def __call__(self, trained_model, eval_data, kwargs) -> float:

        reference_shower = []
        reference_energy = []
        for energy, _, data in eval_data: 
            reference_shower.append(data)
            reference_energy.append(energy)

        reference_shower = np.concatenate(reference_shower)
        reference_energy = np.concatenate(reference_energy)

        generated, energies = trained_model.generate(
            data_loader=eval_data, 
            sample_steps=trained_model.config.get("NSTEPS"), 
            sample_offset=0
        )

        self.hlf.CalculateFeatures(generated)
        self.reference_hlf.CalculateFeatures(reference_shower)
        
        source_array = self.pre_process(energies, self.hlf, 0.)[:, :-1]
        reference_array = self.pre_process(reference_energy, self.reference_hlf, 1.)[:, :-1]
        
        try: 
            fpd, _ = jetnet.evaluation.fpd(
                np.nan_to_num(source_array), np.nan_to_num(reference_array)
            )
        except ValueError as err:
            raise FDPCalculationError(err)

        return fpd


class ComparisonNetwork(ResNet): 
    def __init__(self, dataset_num: Literal[2, 3]):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.inplanes = 15
        dataset_size = {
            2: (-1, 45, 16, 9), 
            3: (-1, 45, 50, 18)
        }
        if dataset_num not in dataset_size.keys(): 
            raise ValueError(f"Only datasets {dataset_size.keys()} can be evaluated with CNNCompare.")

        self.dataset_size = dataset_size[dataset_num]

        self.input_conv = torch.nn.Conv2d(45, 32,
            kernel_size=3, 
            stride=2) # kernel 7, stride 2
        self.local_pool = torch.nn.MaxPool3d(kernel_size=3, stride=2)

        # ResNet18
        self.layer1 = self._make_layer(BasicBlock, 32, blocks=2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 96, blocks=2, stride=1)
        self.layer4 = self._make_layer(BasicBlock, 128, blocks=2, stride=1)

        # concat with energy, apply a batch normalization layer 
        self.batch_norm = torch.nn.BatchNorm1d(3)
        self.fcl = torch.nn.Linear(258, 1)

    def forward(self, x, E): 
        # Reshape into the input shape
        x = x.reshape(self.dataset_size)

        x = self.input_conv(x)
        x = self.local_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, start_dim=1)

        # Append Energy
        x = torch.cat([x, E], axis=-1)

        reshape = 3
        x = x.reshape((-1, reshape,  int(x.shape[1]/reshape)))
        x = self.batch_norm(x).flatten()
        x = self.fcl(x)

        return x

class CNNCompare: 
    """Calculate the log postieor of a model output"""
    def __init__(self, trained_model, config, flags):
        self.flags = flags
        self.config = config
        self.trained_model = trained_model
        self.device = self.trained_model.device
        self.tqdm = utils.import_tqdm()

        if hasattr(self.config["flags"], "sample_offset"): 
            self.sample_offset = self.config["flags"].sample_offset,
        else: 
            self.sample_offset = 0 

        self.cnn = self.load_network()

        if self.config.get("RETRAIN_EVAL_NETWORK", False): 
            self.cnn = self.train_network()


    def load_network(self): 
        base_path =  self.config['flags'].results_folder
        pretrained_weights = f"{base_path.rstrip('/')}/{self.config.get('EVAL_NETWORK', 'eval_cnn')}.pt"
        network = ComparisonNetwork(self.config.get("DATASET_NUM"))

        try: 
            state_dict = torch.load(pretrained_weights, weights_only=True)
            network.load_state_dict(state_dict)
        except FileNotFoundError: 
            print(f"WARNING: Cannot find weights at path {pretrained_weights}")

        return network

    def train_network(self): 
        num = self.config.get("DATASET_NUM")
        training_data = utils.load_data(self.flags, self.config, eval=True)

        config = {
            2: {
                "epochs": 48, 
                "lr": 2.5 * 10e-5
            }, 
            3: {
                "epochs": 12, 
                "lr": 5e-5
            }
        }
        if num not in config.keys(): 
            raise ValueError(f"Dataset #{num} not implemented")

        config = config[num]
        network = ComparisonNetwork(num)
        optimizer = torch.optim.AdamW(network.parameters(), lr=config['lr'])

        for _ in self.tqdm(range(config["epochs"]), "training model ..."): 
            for _, (E, layers, data) in enumerate(training_data):
                E = E.to(device=self.device)
                data = data.to(device=self.device)

                self.cnn.zero_grad()
                optimizer.zero_grad()

                batch_generated = self.trained_model.sample(
                    E,
                    layers=layers,
                    num_steps=self.config["NSTEPS"], # MUST have n_steps
                    debug=False,
                    sample_offset=self.sample_offset,
                )

                p_true = self.cnn.forward(data, E)
                p_predict = self.cnn.forward(batch_generated, E)

                loss = 1 - torch.nn.CrossEntropyLoss()(p_true, p_predict)

                loss.backward()
                optimizer.step()
        
        base_path =  self.config['flags'].results_folder
        pretrained_weights = f"{base_path.rstrip('/')}/{self.config.get('EVAL_NETWORK', 'eval_cnn')}.pt"
        torch.save(self.cnn.state_dict(), pretrained_weights)

        return self.cnn

    def __call__(self, eval_data):
        probabilities = []
        for _, (E, layers, data) in self.tqdm(enumerate(eval_data), "Evaluating..."):
            E = E.to(device=self.device)
            data = data.to(device=self.device)

            batch_generated = self.trained_model.sample(
                E,
                layers=layers,
                num_steps=self.config["NSTEPS"], # MUST have n_steps,
                debug=False,
                sample_offset=self.sample_offset,
            )

            probabilities.append(self.cnn.forward(torch.tensor(batch_generated), E).detach())

        return (1/len(probabilities))*np.sum(np.log(np.array(probabilities)))
