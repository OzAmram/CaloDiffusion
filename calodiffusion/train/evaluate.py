"""

General evaluation metrics for a fully trained model (not losses)

"""
from typing import Literal
from calodiffusion.utils import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch 
from torch.utils import data as torchdata

from torchvision.models.resnet import ResNet, BasicBlock

import calodiffusion.utils.HighLevelFeatures as HLF
from calodiffusion.utils.HGCal_utils import HighLevelFeatures as HGCAL_HLF
from calodiffusion.utils import HGCal_utils as hgcal_utils

try: 
    import jetnet
except ImportError:
    jetnet = None

class FPDCalculationError(Exception): 
    def __init__(self, *args):
        super().__init__(*args)


class _FPD_HGCAL():
    def __init__(self, binning_dataset):
        self.hlf = HGCAL_HLF(binning_dataset)  # Hgcal HLF is stateless(ish), we don't need one for reference 

    def __call__(self, eval_data, trained_model, *args, **kwds):
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
        source = self.hlf(generated, energies)
        reference = self.hlf(reference_shower, reference_energy)

        return {"source": source, "reference": reference}


class _FPD(): 
    def __init__(self, particle, binning_dataset):
        self.hlf = HLF.HighLevelFeatures(particle, binning_dataset)
        self.reference_hlf = HLF.HighLevelFeatures(particle, binning_dataset)
    
    def pre_process(self, energies, hlf_class, label):
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
    
    def __call__(self, eval_data, trained_model, *args, **kwds):
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

        return {"source": source_array, "reference": reference_array}

class FPD: 
    def __init__(self, binning_dataset, particle, hgcal=False): 
        if jetnet is None: 
            raise ImportError("jetnet is not installed. Please install it to use FPD evaluation.")
        if hgcal:
            self.fpd = _FPD_HGCAL(binning_dataset)
        else: 
            self.fpd = _FPD(particle, binning_dataset)

    def __call__(self, trained_model, eval_data, kwargs) -> float:
        out = self.fpd(eval_data, trained_model, **kwargs)
        source_array = out["source"]
        reference_array = out["reference"]

        try: 
            fpd, _ = jetnet.evaluation.fpd(
                np.nan_to_num(source_array), np.nan_to_num(reference_array)
            )
        except ValueError as err:
            raise FPDCalculationError(err)

        return fpd


class ComparisonNetwork(ResNet): 
    def __init__(self, dataset_num: Literal[2, 3, 111]):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.inplanes = 15

        dataset_size = {
            2: (-1, 45, 16, 9), 
            3: (-1, 45, 50, 18), 
            111: (-1, 49, 16, 9), 
        }  
        if dataset_num not in dataset_size.keys(): 
            raise ValueError(f"Only datasets {dataset_size.keys()} can be evaluated with CNNCompare.")

        self.dataset_num = dataset_num
        self.dataset_size = dataset_size[dataset_num]

        if dataset_num in [2, 3]: 
            self.input_conv = torch.nn.Sequential(
                torch.nn.Conv2d(45, 32, kernel_size=3, stride=2),
                torch.nn.MaxPool3d(kernel_size=3, stride=2),
            )

        else: # dataset_num == 111
            self.input_conv = torch.nn.Sequential(
                torch.nn.Conv2d(49, 32, kernel_size=3, stride=2), 
                torch.nn.MaxPool3d(kernel_size=3, stride=2),
            )

        self.layer1 = self._make_layer(BasicBlock, 32, blocks=2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 96, blocks=2, stride=1)
        self.layer4 = self._make_layer(BasicBlock, 128, blocks=2, stride=1)

        # concat with energy, apply a batch normalization layer 
        if dataset_num in [2, 3]:
            self.fcl = torch.nn.Sequential(
                torch.nn.BatchNorm1d(3),
                torch.nn.Linear(258, 1)
            )
        else: 
            self.fcl = torch.nn.Sequential(
                torch.nn.Linear(131, 1)
            )

    def forward(self, x, E): 
        # Reshape into the input shape
        x = x.reshape(self.dataset_size)

        x = self.input_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, start_dim=1)

        # Append Energy
        x = torch.cat([x, E], axis=-1)
        if not self.dataset_num == 111:
            reshape = 3 
            x = x.reshape((-1, reshape,  int(x.shape[1]/reshape)))
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

        if hasattr(self.config["flags"], "sample_offset") and (self.config['flags'].get("sample_offset") is not None): 
            self.sample_offset = self.config["flags"].sample_offset
        else: 
            self.sample_offset = 0 

        self.cnn = self.load_network()

        if self.config.get("RETRAIN_EVAL_NETWORK", False): 
            self.cnn = self.train_network()


    def load_network(self): 
        base_path =  self.config.get("EVAL_NETWORK_PATH", "./calodiffusion/utils")
        pretrained_weights = f"{base_path.rstrip('/')}/{self.config.get('EVAL_NETWORK', 'eval_cnn')}.pt"
        network = ComparisonNetwork(self.config.get("DATASET_NUM")).to(device=self.device)

        try: 
            state_dict = torch.load(pretrained_weights, weights_only=True, map_location=self.device)
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
            }, 
            111: {  # TODO Verify these are fine numbers - nothing in the og paper
                "epochs": 32, 
                "lr": 1e-4
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
        for E, layers, data in self.tqdm(eval_data):
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


class HistogramSeparation: 
    """Make a histogram of a given metric given data and prediction, and then calculate the difference"""
    def __init__(self, metric, bin_file=None, data_shape=[], config_settings={}):
        self.plotter = []        
        if isinstance(metric, str):
            metric = [metric]

        config_settings.update(dict(SHAPE_FINAL=data_shape, BIN_FILE=bin_file))
        for m in metric:
            plotter = utils.load_attr("plot", m)(
                utils.dotdict(), 
                config_settings
            )
            if not hasattr(plotter, "_separation_power"):
                raise TypeError(f"The loaded plotter must be a subclass of 'Histogram', but got {type(plotter).__name__}.")
            
            self.plotter.append(plotter)

    def _single_metric(self, plotter_engine, original, generated, energies): 
        if isinstance(original, torch.Tensor):
            original = original.detach().numpy()
        if isinstance(generated, torch.Tensor):
            generated = generated.detach().numpy()
        if isinstance(energies, torch.Tensor):
            energies = energies.detach().numpy()

        feed_dict = plotter_engine.transform_data(
            {"Geant4": original, "gen": generated}, energies
        )

        def calc_separation(feed_dict):
            original = feed_dict["Geant4"]
            generated = feed_dict["gen"]
            binning = plotter_engine.produce_binning(original)

            histogram_og, _ = np.histogram(original, bins=binning, density=True)
            histogram_generated, _ = np.histogram(generated, bins=binning, density=True)

            return plotter_engine._separation_power(histogram_generated, histogram_og, binning)
        
        if isinstance(feed_dict, dict): 
            metrics = []
            histograms = {key: value for key, value in feed_dict.items() if "hist" in key}
            for key, value in histograms.items(): 
                metrics.append(calc_separation(value))
            return np.mean(metrics)
        
        else: 
            return calc_separation(feed_dict)
    
    def __call__(self, original, generated, energies, *args, **kwds):
        metrics = []
        for plot in self.plotter: 
            metrics.append(self._single_metric(plot, original, generated, energies))
        
        return np.mean(metrics)


class GenericHistogramSeparation(HistogramSeparation): 
    def __init__(self, bin_file=None, data_shape=[], config_settings={}):
        super().__init__("HistERatio", bin_file, data_shape, config_settings)

    def calc_separation(self, original, generated, energies, binning=None, normalize=True): 
        
        if binning is None: 
            binning = np.linspace( np.quantile(original,0.0),np.quantile(original,1),50)

        histogram_og, _ = np.histogram(original, bins=binning, density=normalize)
        histogram_generated, _ = np.histogram(generated, bins=binning, density=normalize)

        return self.plotter[0]._separation_power(histogram_generated, histogram_og, binning)

    def __call__(self, original, generated, energies, *args, **kwds):
        return self.calc_separation(original, generated, energies, *args, **kwds)
    

class DNN(torch.nn.Module):
    """ NN for vanilla classifier. Does not have sigmoid activation in last layer, should
        be used with torch.nn.BCEWithLogitsLoss()
    """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """ Forward pass through the DNN """
        x = self.layers(x)
        return x

class DNNCompare: 
    def __init__(self, input_shape, num_layers=2, num_hidden=2024, dropout_probability=0.2, batch_size=32, n_training_iters=5, n_epochs=10):

    
        self.classifier = DNN(input_dim = input_shape, num_layer= num_layers, num_hidden = num_hidden, dropout_probability = dropout_probability)
        self.classifier.to(utils.get_device())

        self.device = utils.get_device()
        self.optimizer = hgcal_utils.TrainDNNCompare()
        self.batch_size = batch_size
        self.n_iters = n_training_iters
        self.n_epochs = n_epochs

    def __call__(self, original, generated, *args, **kwds):
        ""
        # Train the DNN from utils

        original = original.reshape(original.shape[0], -1)
        generated = generated.reshape(generated.shape[0], -1)
        labels_generated = np.ones((generated.shape[0], 1), dtype=np.float32)
        labels_original = np.zeros((original.shape[0], 1), dtype=np.float32)

        labels_all = np.concatenate((labels_generated, labels_original), axis = 0)
        feats_all = np.concatenate((generated, original), axis = 0)

        scaler = StandardScaler()
        feats_all = scaler.fit_transform(feats_all)
        inputs_all = np.concatenate((feats_all, labels_all), axis = 1, dtype=np.float32)

        train_data, placeholder = utils.split_data_np(inputs_all, 0.7)
        test_data, val_data = utils.split_data_np(placeholder, 0.4)

        cls_lr = 1e-4
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr= cls_lr)

        train_data = torchdata.TensorDataset(torch.tensor(train_data).to(self.device))
        test_data = torchdata.TensorDataset(torch.tensor(test_data).to(self.device))
        val_data = torchdata.TensorDataset(torch.tensor(val_data).to(self.device))

        train_dataloader = torchdata.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_dataloader = torchdata.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        val_dataloader = torchdata.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        acc, auc, JSD = [], [], []
        for _ in range(self.n_iters):
            classifier = self.optimizer.train_and_evaluate_cls(self.classifier, train_dataloader, val_dataloader, optimizer, self.n_epochs)

            with torch.no_grad():
                print("Now looking at independent dataset:")
                eval_acc, eval_auc, eval_JSD = self.optimizer.evaluate_cls(
                    classifier, 
                    test_dataloader,
                    final_eval=True,
                    calibration_data=val_dataloader
                )
            acc.append(eval_acc)
            auc.append(eval_auc)
            JSD.append(eval_JSD)

        return {"ACC": np.mean(acc), "AUC": np.mean(auc), "JSD": np.mean(JSD)}