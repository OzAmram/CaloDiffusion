from calodiffusion.utils.XMLHandler import XMLHandler
from calodiffusion.utils.dataset import Dataset
import calodiffusion.utils.HGCal_utils as HGCal_utils
import calodiffusion.utils.consts as constants

import os
from typing import Literal
import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import sys
import joblib
import matplotlib.pyplot as plt
import torch.utils.data as torchdata


def import_tqdm(): 
    if sys.stderr.isatty():
        from tqdm import tqdm
    else:
        def tqdm(iterable, **kwargs):
            return iterable
    return tqdm
        
def split_data_np(data, frac=0.8):
    np.random.shuffle(data)
    split = int(frac * data.shape[0])
    train_data = data[:split]
    test_data = data[split:]
    return train_data, test_data


def create_phi_image(device, shape=(1, 45, 16, 9)):
    n_phi = shape[-2]
    phi_bins = torch.linspace(0.0, 1.0, n_phi, dtype=torch.float32)
    phi_image = torch.zeros(shape, device=device)
    for i in range(n_phi):
        phi_image[:, :, i, :] = phi_bins[i]
    return phi_image


def create_R_Z_image(device, dataset_num=1, scaled=True, shape=(1, 45, 16, 9)):
    if dataset_num == 0:  # dataset 1, pions
        r_bins = [
            0.00,
            1.00,
            4.00,
            5.00,
            7.00,
            10.00,
            15.00,
            20.00,
            30.00,
            50.00,
            80.00,
            90.00,
            100.00,
            130.00,
            150.00,
            160.00,
            200.00,
            250.00,
            300.00,
            350.00,
            400.00,
            600.00,
            1000.00,
            2000.00,
        ]
    elif dataset_num == 1:  # dataset 1, photons
        r_bins = [
            0.0,
            2.0,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
            120.0,
            130.0,
            150.0,
            160.0,
            200.0,
            250.0,
            300.0,
            350.0,
            400.0,
            600.0,
            1000.0,
            2000.0,
        ]
    elif dataset_num == 2:  # dataset 2
        r_bins = [0, 4.65, 9.3, 13.95, 18.6, 23.25, 27.9, 32.55, 37.2, 41.85]
    elif dataset_num == 3:  # dataset 3
        r_bins = [
            0,
            2.325,
            4.65,
            6.975,
            9.3,
            11.625,
            13.95,
            16.275,
            18.6,
            20.925,
            23.25,
            25.575,
            27.9,
            30.225,
            32.55,
            34.875,
            37.2,
            39.525,
            41.85,
        ]
    elif dataset_num >= 100:  # HGCal
        r_bins = torch.arange(0, shape[-1] + 1)
    else:
        print("RZ binning missing for dataset num %i ? " % (dataset_num))

    r_avgs = [(r_bins[i] + r_bins[i + 1]) / 2.0 for i in range(len(r_bins) - 1)]

    if len(r_avgs) != shape[-1]: 
        raise ValueError(f"Mismatch for dataset size {shape} and dataset num {dataset_num} - expecting dataset with final dim {len(r_avgs)}")
    Z_image = torch.zeros(shape, device=device)
    R_image = torch.zeros(shape, device=device)
    for z in range(shape[1]):
        Z_image[:, z, :, :] = z

    for r in range(shape[-1]):
        R_image[:, :, :, r] = r_avgs[r]
    if scaled:
        r_max = r_avgs[-1]
        z_max = shape[1]
        Z_image /= z_max
        R_image /= r_max
    return R_image, Z_image


def split_data(data, nevts, frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac * nevts)).repeat()
    test_data = data.skip(int(frac * nevts)).repeat()
    return train_data, test_data


def name_translate(generated_file_path:str): 
    try: 
        return generated_file_path.split('/')[-2].split('_')[-1]
    except IndexError: 
        return "generated"


def _separation_power(hist1, hist2, bins):
    """computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
    Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
    plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1 * np.diff(bins), hist2 * np.diff(bins)
    ret = (hist1 - hist2) ** 2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()


def make_histogram(
    entries,
    labels,
    colors,
    xaxis_label="",
    title="",
    num_bins=10,
    logy=False,
    normalize=False,
    stacked=False,
    h_type="step",
    h_range=None,
    fontsize=16,
    fname="",
    yaxis_label="",
    ymax=-1,
):
    alpha = 1.0
    if stacked:
        h_type = "barstacked"
        alpha = 0.2
    fig_size = (8, 6)
    fig = plt.figure(figsize=fig_size)
    ns, bins, patches = plt.hist(
        entries,
        bins=num_bins,
        range=h_range,
        color=colors,
        alpha=alpha,
        label=labels,
        density=normalize,
        histtype=h_type,
    )
    plt.xlabel(xaxis_label, fontsize=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize)

    if logy:
        plt.yscale("log")
    elif ymax > 0:
        plt.ylim([0, ymax])
    else:
        ymax = 1.3 * np.amax(ns)
        plt.ylim([0, ymax])

    if yaxis_label != "":
        plt.ylabel(yaxis_label, fontsize=fontsize)
        plt.tick_params(axis="y", labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc="upper right", fontsize=fontsize)
    if fname != "":
        plt.savefig(fname)
        print("saving fig %s" % fname)
    # else: plt.show(block=False)
    return fig

def reverse_logit(x, alpha=1e-6):
    exp = np.exp(x)
    o = exp / (1 + exp)
    o = (o - alpha) / (1 - 2 * alpha)
    return o


def logit(x, alpha=1e-6):
    o = alpha + (1 - 2 * alpha) * x
    o = np.ma.log(o / (1 - o)).filled(0)
    return o


def DataLoader(file_name, hgcal=False, **kwargs):
    if hgcal:
        return HGCal_utils.DataLoaderHGCal(file_name, **kwargs)
    else:
        return DataLoaderCaloChall(file_name, **kwargs)


def ReverseNorm(voxels, e, hgcal=False, **kwargs):
    if hgcal:
        return HGCal_utils.ReverseNormHGCal(voxels, e, **kwargs)
    else:
        return ReverseNormCaloChall(voxels, e, **kwargs)


def DataLoaderCaloChall(
    file_name,
    shape=None,
    emax=99999.0,
    emin=0.0001,
    binning_file="",
    nevts=-1,
    max_deposit=2,
    ecut=0,
    logE=True,
    showerMap="log-norm",
    nholdout=0,
    from_end=False,
    dataset_num=2,
    orig_shape=False,
    evt_start=0,
    shower_scale=0.001,
    **kwargs,
):
    with h5.File(file_name, "r") as h5f:
        # holdout events for testing
        if nevts == -1 and nholdout > 0:
            nevts = -(nholdout)
        end = evt_start + int(nevts)
        if from_end:
            evt_start = -int(nevts)
            end = None
        if end == -1:
            end = None
        print("Event start, stop: ", evt_start, end)
        e = h5f["incident_energies"][evt_start:end].astype(np.float32) * shower_scale
        shower = h5f["showers"][evt_start:end].astype(np.float32) * shower_scale

    e = np.reshape(e, (-1, 1))

    shower_preprocessed, layerE_preprocessed = preprocess_shower(
        shower,
        e,
        shape,
        binning_file,
        showerMap,
        dataset_num=dataset_num,
        orig_shape=orig_shape,
        ecut=ecut,
        max_deposit=max_deposit,
    )

    if logE:
        E_preprocessed = np.log10(e / emin) / np.log10(emax / emin)
    else:
        E_preprocessed = (e - emin) / (emax - emin)

    return shower_preprocessed, E_preprocessed, layerE_preprocessed


def preprocess_shower(
    shower,
    e,
    shape,
    binning_file,
    showerMap="log-norm",
    dataset_num=2,
    orig_shape=False,
    ecut=0,
    max_deposit=2,
):
    if dataset_num == 1:
        bins = XMLHandler("photon", binning_file)
    elif dataset_num == 0:
        bins = XMLHandler("pion", binning_file)

    if dataset_num <= 1 and not orig_shape:
        g = GeomConverter(bins)
        shower = g.convert(g.reshape(shower))
    elif not orig_shape:
        shower = shower.reshape(shape)

    if dataset_num > 3 or dataset_num < 0:
        print("Invalid dataset %i!" % dataset_num)
        exit(1)

    if orig_shape and dataset_num <= 1:
        dataset_num += 10

    print("dset", dataset_num)

    c = constants.dataset_params[dataset_num]

    if "quantile" in showerMap and ecut > 0:
        np.random.seed(123)
        noise = (ecut / 3) * np.random.rand(*shower.shape)
        shower += noise

    alpha = 1e-6
    per_layer_norm = False

    layerE = None
    prefix = ""
    if "layer" in showerMap:
        eshape = (-1, *(1,) * (len(shower.shape) - 1))
        shower = np.ma.divide(shower, (max_deposit * e.reshape(eshape)))
        # regress total deposited energy and fraction in each layer
        if dataset_num % 10 > 1 or not orig_shape:
            layers = np.sum(shower, (3, 4), keepdims=True)
            totalE = np.sum(shower, (2, 3, 4), keepdims=True)
            if per_layer_norm:
                shower = np.ma.divide(shower, layers)
            shower = np.reshape(shower, (shower.shape[0], -1))

        else:
            # use XML handler to deal with irregular binning of layers for dataset 1
            boundaries = np.unique(bins.GetBinEdges())
            layers = np.zeros(
                (shower.shape[0], boundaries.shape[0] - 1), dtype=np.float32
            )

            totalE = np.sum(shower, 1, keepdims=True)
            for idx in range(boundaries.shape[0] - 1):
                layers[:, idx] = np.sum(
                    shower[:, boundaries[idx] : boundaries[idx + 1]], 1
                )
                if per_layer_norm:
                    shower[:, boundaries[idx] : boundaries[idx + 1]] = np.ma.divide(
                        shower[:, boundaries[idx] : boundaries[idx + 1]],
                        layers[:, idx : idx + 1],
                    )

        # only logit transform for layers
        layer_alpha = 1e-6
        layers = np.ma.divide(layers, totalE)
        layers = logit(layers)

        layers = (layers - c["layers_mean"]) / c["layers_std"]
        totalE = (totalE - c["totalE_mean"]) / c["totalE_std"]
        # append totalE to layerE array
        totalE = np.reshape(totalE, (totalE.shape[0], 1))
        layers = np.squeeze(layers)

        layerE = np.concatenate((totalE, layers), axis=1)

        if per_layer_norm:
            prefix = "layerN_"
    else:
        shower = np.reshape(shower, (shower.shape[0], -1))
        shower = shower / (max_deposit * e)

    if "logit" in showerMap:
        shower = logit(shower)

        if "norm" in showerMap:
            shower = (shower - c[prefix + "logit_mean"]) / c[prefix + "logit_std"]
        elif "scaled" in showerMap:
            shower = (
                2.0 * (shower - c["logit_min"]) / (c["logit_max"] - c["logit_min"])
                - 1.0
            )

    elif "log" in showerMap:
        eps = 1e-8
        shower = np.ma.log(shower).filled(c["log_min"])
        if "norm" in showerMap:
            shower = (shower - c[prefix + "log_mean"]) / c[prefix + "log_std"]
        elif "scaled" in showerMap:
            shower = (
                2.0
                * (shower - c[prefix + "log_min"])
                / (c[prefix + "log_max"] - c[prefix + "log_min"])
                - 1.0
            )

    if "quantile" in showerMap and c[prefix + "qt"] is not None:
        print("Loading quantile transform from %s" % c["qt"])
        qt = joblib.load(c["qt"])
        shape = shower.shape
        shower = qt.transform(shower.reshape(-1, 1)).reshape(shower.shape)

    return shower, layerE


def LoadJson(file_name):
    import yaml

    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


def ReverseNormCaloChall(
    voxels,
    e,
    emax=9999.0,
    emin=0.0001,
    config=None,
    shape=None,
    binning_file="",
    max_deposit=2,
    logE=True,
    layerE=None,
    showerMap="log",
    dataset_num=2,
    orig_shape=False,
    ecut=0.0,
    **kwargs,
):
    """Revert the transformations applied to the training set"""

    if dataset_num > 3 or dataset_num < 0:
        print("Invalid dataset %i!" % dataset_num)
        exit(1)

    if dataset_num == 1:
        bins = XMLHandler("photon", binning_file)
    elif dataset_num == 0:
        bins = XMLHandler("pion", binning_file)

    if orig_shape and dataset_num <= 1:
        dataset_num += 10
    print("dset", dataset_num)
    c = constants.dataset_params[dataset_num]

    alpha = 1e-6
    if logE:
        energy = emin * (emax / emin) ** e
    else:
        energy = emin + (emax - emin) * e

    prefix = ""
    # if('layer' in showerMap): prefix = "layerN_"

    if "quantile" in showerMap and c["qt"] is not None:
        print("Loading quantile transform from %s" % c["qt"])
        qt = joblib.load(c["qt"])
        shape = voxels.shape
        voxels = qt.inverse_transform(voxels.reshape(-1, 1)).reshape(shape)

    if "logit" in showerMap:
        if "norm" in showerMap:
            voxels = (voxels * c[prefix + "logit_std"]) + c[prefix + "logit_mean"]
        elif "scaled" in showerMap:
            voxels = (voxels + 1.0) * 0.5 * (
                c[prefix + "logit_max"] - c[prefix + "logit_min"]
            ) + c[prefix + "logit_min"]

        # avoid overflows
        # voxels = np.minimum(voxels, np.log(max_deposit/(1-max_deposit)))

        data = reverse_logit(voxels)

    elif "log" in showerMap:
        if "norm" in showerMap:
            voxels = (voxels * c[prefix + "log_std"]) + c[prefix + "log_mean"]
        elif "scaled" in showerMap:
            voxels = (voxels + 1.0) * 0.5 * (
                c[prefix + "log_max"] - c[prefix + "log_min"]
            ) + c[prefix + "log_min"]

        voxels = np.minimum(voxels, np.log(max_deposit))

        data = np.exp(voxels)

    # Per layer energy normalization
    if "layer" in showerMap:
        assert layerE is not None
        totalE, layers = layerE[:, :1], layerE[:, 1:]
        totalE = (totalE * c["totalE_std"]) + c["totalE_mean"]
        layers = (layers * c["layers_std"]) + c["layers_mean"]

        layers = reverse_logit(layers)

        # scale layer energies to total deposited energy
        layers /= np.sum(layers, axis=1, keepdims=True)
        layers *= totalE

        data = np.squeeze(data)

        # remove voxels with negative energies so they don't mess up sums
        eps = 1e-6
        data[data < 0] = 0
        # layers[layers < 0] = eps

        # Renormalize layer energies
        if dataset_num % 10 > 1 or not orig_shape:
            prev_layers = np.sum(data, (2, 3), keepdims=True)
            layers = layers.reshape((-1, data.shape[1], 1, 1))
            rescale_facs = layers / (prev_layers + 1e-10)
            # If layer is essential zero from base network or layer network, don't rescale
            rescale_facs[layers < eps] = 1.0
            rescale_facs[prev_layers < eps] = 1.0
            data *= rescale_facs
        else:
            boundaries = np.unique(bins.GetBinEdges())
            for idx in range(boundaries.shape[0] - 1):
                prev_layer = np.sum(
                    data[:, boundaries[idx] : boundaries[idx + 1]], 1, keepdims=True
                )
                rescale_fac = layers[:, idx : idx + 1] / (prev_layer + 1e-10)
                rescale_fac[layers[:, idx : idx + 1] < eps] = 1.0
                rescale_fac[prev_layer < eps] = 1.0
                data[:, boundaries[idx] : boundaries[idx + 1]] *= rescale_fac

    if dataset_num > 1 or orig_shape:
        data = data.reshape(voxels.shape[0], -1) * max_deposit * energy.reshape(-1, 1)
    else:
        g = GeomConverter(bins)
        data = np.squeeze(data)
        data = g.unreshape(g.unconvert(data)) * max_deposit * energy.reshape(-1, 1)

    if "quantile" in showerMap and ecut > 0.0:
        # subtact of avg of added noise
        data -= 0.5 * (ecut / 3)

    if ecut > 0:
        data[data < ecut] = 0  # min from samples

    return data, energy


class NNConverter(nn.Module):
    "Convert irregular geometry to regular one, initialized with regular geometric conversion, but uses trainable linear map"

    def __init__(self, geomconverter=None, bins=None, hidden_size=32):
        super().__init__()
        if geomconverter is None:
            geomconverter = GeomConverter(bins)

        self.gc = geomconverter

        self.encs = nn.ModuleList([])
        self.decs = nn.ModuleList([])
        eps = 1e-5
        for i in range(len(self.gc.weight_mats)):
            rdim_in = len(self.gc.lay_r_edges[i]) - 1
            # lay = nn.Sequential(*[nn.Linear(rdim_in, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size),
            #    nn.GELU(), nn.Linear(hidden_size, self.gc.dim_r_out)])

            lay = nn.Linear(rdim_in, self.gc.dim_r_out, bias=False)
            noise = torch.randn_like(self.gc.weight_mats[i])
            lay.weight.data = self.gc.weight_mats[i] + eps * noise

            self.encs.append(lay)

            # inv_lay = nn.Sequential(*[nn.Linear(self.gc.dim_r_out, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size),
            # nn.GELU(), nn.Linear(hidden_size, rdim_in)])
            inv_lay = nn.Linear(self.gc.dim_r_out, rdim_in, bias=False)

            inv_init = torch.linalg.pinv(self.gc.weight_mats[i])
            noise2 = torch.randn_like(inv_init)
            inv_lay.weight.data = inv_init + eps * noise2

            self.decs.append(inv_lay)

    def enc(self, x):
        n_shower = x.shape[0]
        x = self.gc.reshape(x)

        out = torch.zeros(
            (n_shower, 1, self.gc.num_layers, self.gc.alpha_out, self.gc.dim_r_out)
        )
        for i in range(len(x)):
            o = self.encs[i](x[i])
            if self.gc.lay_alphas is not None:
                if self.gc.lay_alphas[i] == 1:
                    # distribute evenly in phi
                    o = (
                        torch.repeat_interleave(o, self.gc.alpha_out, dim=-2)
                        / self.gc.alpha_out
                    )
                elif self.gc.lay_alphas[i] != self.gc.alpha_out:
                    print(
                        "Num alpha bins for layer %i is %i. Don't know how to handle"
                        % (i, self.gc.lay_alphas[i])
                    )
                    exit(1)
            out[:, 0, i] = o
        return out

    def dec(self, x):
        out = []
        x = torch.squeeze(x, dim=1)
        for i in range(self.gc.num_layers):
            o = self.decs[i](x[:, i])

            if self.gc.lay_alphas is not None:
                if self.gc.lay_alphas[i] == 1:
                    # Only works for converting 1 alpha bin into multiple, ok for dataset1 but maybe should generalize
                    o = torch.sum(o, dim=-2, keepdim=True)
                elif self.gc.lay_alphas[i] != self.gc.alpha_out:
                    print(
                        "Num alpha bins for layer %i is %i. Don't know how to handle"
                        % (i, self.gc.lay_alphas[i])
                    )
                    exit(1)
            out.append(o)
        out = self.gc.unreshape(out)
        return out

    def forward(self, x):
        return self.enc(x)


class GeomConverter:
    "Convert irregular geometry to regular one (ala CaloChallenge Dataset 1)"

    def __init__(
        self,
        bins=None,
        all_r_edges=None,
        lay_r_edges=None,
        alpha_out=1,
        lay_alphas=None,
    ):
        self.layer_boundaries = []
        self.bins = None

        # init from binning
        if bins is not None:
            self.layer_boundaries = np.unique(bins.GetBinEdges())
            rel_layers = bins.GetRelevantLayers()
            lay_alphas = [
                len(bins.alphaListPerLayer[idx][0])
                for idx, redge in enumerate(bins.r_edges)
                if len(redge) > 1
            ]
            alpha_out = np.amax(lay_alphas)

            all_r_edges = []

            lay_r_edges = [bins.r_edges[l] for l in rel_layers]
            for ilay in range(len(lay_r_edges)):
                for r_edge in lay_r_edges[ilay]:
                    all_r_edges.append(r_edge)
            all_r_edges = torch.unique(torch.FloatTensor(all_r_edges))

        self.all_r_edges = all_r_edges
        self.lay_r_edges = lay_r_edges
        self.alpha_out = alpha_out
        self.lay_alphas = lay_alphas
        self.num_layers = len(self.lay_r_edges)

        self.all_r_areas = all_r_edges[1:] ** 2 - all_r_edges[:-1] ** 2
        self.dim_r_out = len(all_r_edges) - 1
        self.weight_mats = []
        for ilay in range(len(lay_r_edges)):
            dim_in = len(lay_r_edges[ilay]) - 1
            lay = nn.Linear(dim_in, self.dim_r_out, bias=False)
            weight_mat = torch.zeros((self.dim_r_out, dim_in))
            for ir in range(dim_in):
                o_idx_start = torch.nonzero(
                    self.all_r_edges == self.lay_r_edges[ilay][ir]
                )[0][0]
                o_idx_stop = torch.nonzero(
                    self.all_r_edges == self.lay_r_edges[ilay][ir + 1]
                )[0][0]

                split_idxs = list(range(o_idx_start, o_idx_stop))
                orig_area = (
                    self.lay_r_edges[ilay][ir + 1] ** 2
                    - self.lay_r_edges[ilay][ir] ** 2
                )

                # split proportional to bin area
                weight_mat[split_idxs, ir] = self.all_r_areas[split_idxs] / orig_area

            self.weight_mats.append(weight_mat)

    def reshape(self, raw_shower):
        # convert to jagged array each of shape (N_shower, N_alpha, N_R)
        shower_reshape = []
        for idx in range(len(self.layer_boundaries) - 1):
            data_reshaped = raw_shower[
                :, self.layer_boundaries[idx] : self.layer_boundaries[idx + 1]
            ].reshape(raw_shower.shape[0], int(self.lay_alphas[idx]), -1)
            shower_reshape.append(data_reshaped)
        return shower_reshape

    def unreshape(self, raw_shower):
        # convert jagged back to original flat format
        n_show = raw_shower[0].shape[0]
        out = torch.zeros((n_show, self.layer_boundaries[-1]))
        for idx in range(len(self.layer_boundaries) - 1):
            out[:, self.layer_boundaries[idx] : self.layer_boundaries[idx + 1]] = (
                raw_shower[idx].reshape(n_show, -1)
            )
        return out

    def convert(self, d):
        out = torch.zeros((len(d[0]), self.num_layers, self.alpha_out, self.dim_r_out))
        for i in range(len(d)):
            if not isinstance(d[i], torch.FloatTensor):
                d[i] = torch.FloatTensor(d[i])
            o = torch.einsum("...ij,...j->...i", self.weight_mats[i], d[i])
            if self.lay_alphas is not None:
                if self.lay_alphas[i] == 1:
                    # distribute evenly in phi
                    o = (
                        torch.repeat_interleave(o, self.alpha_out, dim=-2)
                        / self.alpha_out
                    )
                elif self.lay_alphas[i] != self.alpha_out:
                    print(
                        "Num alpha bins for layer %i is %i. Don't know how to handle"
                        % (i, self.lay_alphas[i])
                    )
                    exit(1)
            out[:, i] = o
        return out

    def unconvert(self, d):
        out = []
        for i in range(self.num_layers):
            weight_mat_inv = torch.linalg.pinv(self.weight_mats[i])
            x = torch.FloatTensor(d[:, i])
            o = torch.einsum("...ij,...j->...i", weight_mat_inv, x)

            if self.lay_alphas is not None:
                if self.lay_alphas[i] == 1:
                    # Only works for converting 1 alpha bin into multiple, ok for dataset1 but maybe should generalize
                    o = torch.sum(o, dim=-2, keepdim=True)
                elif self.lay_alphas[i] != self.alpha_out:
                    print(
                        "Num alpha bins for layer %i is %i. Don't know how to handle"
                        % (i, self.lay_alphas[i])
                    )
                    exit(1)
            out.append(o)
        return out


class EarlyStopper:
    def __init__(self, patience=1, mode="loss", min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.mode = mode

    def early_stop(self, var):
        if self.mode == "val_loss":
            validation_loss = var
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        elif self.mode == "diff":
            if var < 0:
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

def draw_shower(shower, dataset_num, fout, title=None):
    from calodiffusion.utils.HighLevelFeatures import HighLevelFeatures

    binning_file = "../CaloChallenge/code/binning_dataset_2.xml"
    hf = HighLevelFeatures("electron", binning_file)
    hf.DrawSingleShower(shower, fout, title=title)


def conversion_preprocess(file_path):
    with h5.File(file_path, "r") as h5f:
        showers = h5f["showers"][:]
    mask = np.sum(showers, 0) == 0
    mask_file = file_path.replace(".hdf5", "_mask.hdf5")
    print("Creating mask file %s " % mask_file)
    with h5.File(mask_file, "w") as h5f:
        h5f.create_dataset("mask", data=mask)


def get_files(field):
    if(isinstance(field, list)):
        return field
    elif(isinstance(field, str)):
        if(not os.path.exists(field)):
            print("File list %s not found" % field)
            return []
        with open(field, "r") as f:
            f_list = [line.strip() for line in f]
            return f_list
    else:
        print("Unrecognized file param ", field)
        return []


def load_data(args, config, eval=False, NN_embed=None):

    nholdout = config.get("HOLDOUT", 0)
    batch_size = config["BATCH"]
    dataset_num = config.get("DATASET_NUM", 2)
    shower_embed = config.get("SHOWER_EMBED", "")
    orig_shape = "orig" in shower_embed
    layer_norm = "layer" in config["SHOWERMAP"]
    geom_file = config.get("BIN_FILE", "")

    pre_embed = "pre-embed" in shower_embed
    shower_scale = config.get("SHOWERSCALE", 200.0)
    max_cells = config.get("MAX_CELLS", None)
    hgcal = config.get("HGCAL", False)

    if eval:
        files = get_files(config["EVAL"])
        val_file_list = []
    else:
        if hasattr(args, "seed"):
            torch.manual_seed(args.seed)
        files = get_files(config["FILES"])
        val_file_list = get_files(config.get("VAL_FILES", []))

    NN_embed = None
    if pre_embed:
        trainable = config.get("TRAINABLE_EMBED", False)
        NN_embed = HGCal_utils.HGCalConverter(
            bins=config["SHAPE_FINAL"],
            geom_file=geom_file,
            trainable=trainable,
            device=get_device(),
        ).to(device=get_device())
        NN_embed.init(norm=True, dataset_num=dataset_num)

    train_files = []
    val_files = []

    n_showers = 0

    for i, dataset in enumerate(files + val_file_list):

        tag = ".npz"
        if args.nevts > 0:
            tag = ".n%i.npz" % args.nevts
        # path of pre-processed data files
        path_clean = os.path.join(args.data_folder, dataset + tag)
        shape = config.get("SHAPE_PAD")
        if shape is None:
            shape = config.get("SHAPE_FINAL")

        if not os.path.exists(path_clean) or args.reclean:
            # process this dataset
            showers, E, layers = DataLoader(
                os.path.join(args.data_folder, dataset),
                shape=shape,
                emax=config["EMAX"],
                emin=config["EMIN"],
                hgcal=hgcal,
                nevts=args.nevts,
                binning_file=geom_file,
                max_deposit=config[
                    "MAXDEP"
                ],  # noise can generate more deposited energy than generated
                logE=config["logE"],
                showerMap=config["SHOWERMAP"],
                shower_scale=shower_scale,
                max_cells=max_cells,
                nholdout=nholdout if (i == len(files) - 1) else 0,
                dataset_num=dataset_num,
                orig_shape=orig_shape,
                config=config,
                embed=pre_embed,
                NN_embed=NN_embed,
            )
            n_showers += showers.shape[0]

            layers = np.reshape(layers, (layers.shape[0], -1))

            if orig_shape:
                showers = np.reshape(showers, config["SHAPE_ORIG"])
            else:
                showers = np.reshape(showers, config["SHAPE_PAD"])

            np.savez_compressed(
                path_clean,
                E=E,
                layers=layers,
                showers=showers,
            )
            del E, layers, showers

        if dataset in files:
            train_files.append(path_clean)
        else:
            val_files.append(path_clean)

        if (args.nevts > 0 and n_showers >= args.nevts):
            break

    dataset_train = Dataset(train_files)
    loader_train = torchdata.DataLoader(
        dataset_train, batch_size=batch_size, pin_memory=True
    )

    loader_val = None
    if len(val_files) > 0:
        dataset_val = Dataset(val_files)
        loader_val = torchdata.DataLoader(
            dataset_val, batch_size=batch_size, pin_memory=True
        )

    return loader_train, loader_val


def subsample_alphas(alpha, time, x_shape):
    batch_size = time.shape[0]
    out = alpha.gather(-1, time.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time.device)


def apply_in_batches(model, data, batch_size=128, device="cpu"):
    data_loader = torchdata.DataLoader(
        torch.Tensor(data), batch_size=batch_size, shuffle=False
    )
    out = None
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        out_ = model(batch)
        if i == 0:
            out = out_.detach().cpu()
        else:
            out = torch.cat([out, out_.detach().cpu()], axis=0)
    return out


def append_h5(f, name, data):
    prev_size = f[name].shape[0]
    f[name].resize((prev_size + data.shape[0]), axis=0)
    f[name][prev_size:] = data


def apply_mask_conserveE(generated, mask):
    # Preserve layer energies after applying a mask
    generated[generated < 0] = 0
    d_masked = np.where(mask, generated, 0.0)
    lostE = np.sum(d_masked, axis=-1, keepdims=True)
    ELayer = np.sum(generated, axis=-1, keepdims=True)
    eps = 1e-10
    rescale = (ELayer + eps) / (ELayer - lostE + eps)
    generated[mask] = 0.0
    generated *= rescale

    return generated

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def subsample_alphas(alpha, time, x_shape):
    batch_size = time.shape[0]
    out = alpha.gather(-1, time.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time.device)


def load_attr(type_: Literal["sampler", "loss"], algo_name: str): 
    if type_ == "sampler": 
        from calodiffusion.models import sample as module
    else: 
        from calodiffusion.models import loss as module

    try: 
        algo = getattr(
            module, algo_name
        )
        
    except AttributeError as e: 
        raise ValueError("%s '%s' is not supported: %s" % (type_, algo_name, e))
    
    return algo
