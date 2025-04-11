from abc import abstractmethod, ABC
import copy
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from matplotlib.colors import LogNorm as LN
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mplhep as hep

from calodiffusion.utils import utils
import calodiffusion.utils.HGCal_utils as HGCal_utils


def WeightedMean(coord, energies, power=1, axis=-1):
    ec = np.sum(energies * np.power(coord, power), axis=axis)
    sum_energies = np.sum(energies, axis=axis)
    ec = np.ma.divide(ec, sum_energies).filled(0)
    return ec


def ang_center_spread(matrix, energies, axis=-1):
    # weighted average over periodic variabel (angle)
    # https://github.com/scipy/scipy/blob/v1.11.1/scipy/stats/_morestats.py#L4614
    # https://en.wikipedia.org/wiki/Directional_statistics#The_fundamental_difference_between_linear_and_circular_statistics
    cos_matrix = np.cos(matrix)
    sin_matrix = np.sin(matrix)
    cos_ec = WeightedMean(cos_matrix, energies, axis=axis)
    sin_ec = WeightedMean(sin_matrix, energies, axis=axis)
    ang_mean = np.arctan2(sin_ec, cos_ec)
    R = np.sqrt(sin_ec**2 + cos_ec**2)
    eps = 1e-8
    R = np.clip(R, eps, 1.0)

    ang_std = np.sqrt(-np.log(R))
    return ang_mean, ang_std


def GetWidth(mean, mean2):
    width = np.ma.sqrt(mean2 - mean**2).filled(0)
    return width


class ScalarFormatterClass(mtick.ScalarFormatter):
    # https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


class Plot(ABC):
    def __init__(self, flags, config) -> None:

        self.flags = flags
        self.config = config

        self.plt_exts = flags.plot_extensions
        self.axis_scales = ["", "_logy"]

        self.line_style = {
            "Geant4": "dotted",
            "Geant4 (CMSSW)": "dotted",
            "CaloDiffusion": "-",
            "HGCaloDiffusion": "-",
            "Avg Shower": "-",
        }

        self.colors = {
            "Geant4": "black",
            "Geant4 (CMSSW)": "black",
            "Avg Shower": "blue",
            "CaloDiffusion": "blue",
            "HGCaloDiffusion": "blue",
        }

        Plot.set_style()
        self.geant_key = "Geant4"

        self.hgcal = config.get("HGCAL", False)

        if (not self.hgcal) or flags.plot_reshape:
            self.shape_plot = config["SHAPE_FINAL"]
        else:
            self.shape_plot = config["SHAPE_PAD"]

    def save_names(self, plot_name) -> list[str]: 
        plot_dir = os.path.join(self.flags.plot_folder, self.config['CHECKPOINT_NAME'])
        os.makedirs(plot_dir, exist_ok=True)

        return [
            os.path.join(plot_dir, f"{plot_name}_{utils.name_translate(self.flags.generated)}{axis_scale}.{extension}")
            for extension in self.plt_exts
            for axis_scale in self.axis_scales
        ]

    def save_fig(self, name, fig, ax0) -> None:
        if ("logy") in name:
            ax0.set_yscale("log")
        else:
            ax0.set_yscale("linear")
        fig.savefig(name)

    @staticmethod
    def set_style():
        from matplotlib import rc

        rc("text", usetex=True)

        import matplotlib as mpl

        rc("font", family="serif")
        rc("font", size=22)
        rc("xtick", labelsize=15)
        rc("ytick", labelsize=15)
        rc("legend", fontsize=24)

        mpl.rcParams.update({"font.size": 26})
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams.update({"xtick.major.size": 8})
        mpl.rcParams.update({"xtick.major.width": 1.5})
        mpl.rcParams.update({"xtick.minor.size": 4})
        mpl.rcParams.update({"xtick.minor.width": 0.8})
        mpl.rcParams.update({"ytick.major.size": 8})
        mpl.rcParams.update({"ytick.major.width": 1.5})
        mpl.rcParams.update({"ytick.minor.size": 4})
        mpl.rcParams.update({"ytick.minor.width": 0.8})

        mpl.rcParams.update({"xtick.labelsize": 18})
        mpl.rcParams.update({"ytick.labelsize": 18})
        mpl.rcParams.update({"axes.labelsize": 26})
        mpl.rcParams.update({"legend.frameon": False})
        mpl.rcParams.update({"lines.linewidth": 4})

    def _plot(
        self,
        feed_dict,
        xlabel="",
        ylabel="",
        reference_name="Geant4",
        no_mean=False,
    ):
        if reference_name not in feed_dict.keys(): 
            raise NotImplementedError("Reference distribution %s not included, choice from %s", (reference_name, feed_dict.keys()))

        fig, gs = self.SetGrid()
        ax0 = plt.subplot(gs[0])
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1], sharex=ax0)

        if self.flags.cms:
            hep.style.use(hep.style.CMS)
            hep.cms.text(ax=ax0, text="Simulation Preliminary")

        for ip, plot in enumerate(feed_dict.keys()):
            color = self.colors.get(plot, "blue")
            linestyle = self.line_style.get(plot, "-")

            if no_mean:
                d = feed_dict[plot]
                ref = feed_dict[reference_name]
            else:
                d = np.mean(feed_dict[plot], 0)
                ref = np.mean(feed_dict[reference_name], 0)
            if "steps" in plot or "r=" in plot:
                ax0.plot(d, label=plot, marker=linestyle, color=color, lw=0)
            else:
                ax0.plot(d, label=plot, linestyle=linestyle, color=color)
            if len(self.flags.plot_label) > 0:
                ax0.set_title(
                    self.flags.plot_label, fontsize=20, loc="right", style="italic"
                )
            if reference_name != plot:
                ax0.get_xaxis().set_visible(False)
                ax0.set_ymargin(0)

                eps = 1e-8
                ratio = np.divide(d, ref + eps)
                # ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)

                plt.axhline(y=1.0, color="black", linestyle="--", linewidth=2)

                if "steps" in plot or "r=" in plot:
                    ax1.plot(
                        ratio,
                        color=color,
                        markeredgewidth=4,
                        marker=linestyle,
                        lw=0,
                    )
                else:
                    ax1.plot(ratio, color=color, linestyle=linestyle)

        self.FormatFig(xlabel="", ylabel=ylabel, ax0=ax0)
        ax0.legend(
            loc="best",
            fontsize=24,
            ncol=1,
            facecolor="white",
            framealpha=0.5,
            frameon=True,
        )

        plt.ylabel("Ratio")
        plt.xlabel(xlabel)
        loc = mtick.MultipleLocator(base=10.0)
        ax1.yaxis.set_minor_locator(loc)
        plt.ylim([0.5, 1.5])

        plt.subplots_adjust(
            left=0.2, right=0.9, top=0.94, bottom=0.12, wspace=0, hspace=0
        )
        # plt.tight_layout()

        return fig, ax0

    @staticmethod
    def SetFig(xlabel, ylabel):
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position("both")
        ax0.xaxis.set_ticks_position("both")
        ax0.tick_params(direction="in", which="both")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel(ylabel, fontsize=24)

        ax0.minorticks_on()
        return fig, ax0

    @staticmethod
    def WriteText(xpos, ypos, text, ax0):
        plt.text(
            xpos,
            ypos,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax0.transAxes,
            fontsize=25,
            fontweight="bold",
        )

    def _separation_power(self, hist1, hist2, bins):
        """computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
        """
        hist1, hist2 = hist1 * np.diff(bins), hist2 * np.diff(bins)
        ret = (hist1 - hist2) ** 2
        ret /= hist1 + hist2 + 1e-16
        return 0.5 * ret.sum()

    def SetGrid(self, ratio=True):
        fig = plt.figure(figsize=(9, 9))
        if ratio:
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            gs.update(wspace=0.025, hspace=0.1)
        else:
            gs = gridspec.GridSpec(1, 1)
        return fig, gs

    def FormatFig(self, xlabel, ylabel, ax0):
        # Limit number of digits in ticks
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel(ylabel, labelpad=10)

    @abstractmethod
    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        raise NotImplementedError

class Histogram(Plot, ABC): 
    """A subclass of Plot specifically for histograms"""
    def produce_binning(self, reference_data): 
        binning = np.linspace(
            np.quantile(reference_data, 0.0),
            np.quantile(reference_data, 1),
            10
        )
        return binning

    @abstractmethod
    def transform_data(self, feed_dict:dict, energies: np.ndarray): 
        """
        Transform the feed dictionary into the variable to be plotted. Required.

        Args:
            feed_dict: Data to transform
            energies: Energy corresponding to data
        """
        raise NotImplementedError

    def _hist(
        self,
        feed_dict,
        xlabel="",
        ylabel="Arbitrary units",
        reference_name="Geant4",
        binning=None,
        label_loc="best",
        ratio=True,
        normalize=True,
        leg_font=24,
    ):
        if reference_name not in feed_dict.keys():
            reference_name = list(feed_dict.keys())[0]
            print("taking %s as ref" % reference_name)

        fig, gs = self.SetGrid(ratio)
        ax0 = plt.subplot(gs[0])
        if ratio:
            plt.xticks(fontsize=0)
            ax1 = plt.subplot(gs[1], sharex=ax0)

        if self.flags.cms:
            hep.style.use(hep.style.CMS)
            hep.cms.text(ax=ax0, text="Simulation Preliminary")

        if binning is None:
            binning = self.produce_binning(feed_dict[reference_name])
        xaxis = [(binning[i] + binning[i + 1]) / 2.0 for i in range(len(binning) - 1)]
        reference_hist, _ = np.histogram(
            feed_dict[reference_name], bins=binning, density=True
        )

        for ip, plot in enumerate(reversed(list(feed_dict.keys()))):
            color = self.colors.get(plot, "blue")
            linestyle = self.line_style.get(plot, "-")

            if "steps" in plot or "r=" in plot:
                dist, _ = np.histogram(feed_dict[plot], bins=binning, density=normalize)
                ax0.plot(
                    xaxis,
                    dist,
                    histtype="stepfilled",
                    facecolor="silver",
                    lw=2,
                    label=plot,
                    alpha=1.0,
                )

            elif "Geant" in plot:
                dist, _, _ = ax0.hist(
                    feed_dict[plot],
                    bins=binning,
                    label=plot,
                    density=True,
                    histtype="stepfilled",
                    facecolor="silver",
                    lw=2,
                    alpha=1.0,
                )
            else:
                dist, _, _ = ax0.hist(
                    feed_dict[plot],
                    bins=binning,
                    label=plot,
                    linestyle=linestyle,
                    color=color,
                    density=True,
                    histtype="step",
                    lw=4,
                )

            if len(self.flags.plot_label) > 0:
                ax0.set_title(
                    self.flags.plot_label, fontsize=20, loc="right", style="italic"
                )

            if reference_name != plot and ratio:
                eps = 1e-8
                h_ratio = np.divide(dist, reference_hist + eps)
                if "steps" in plot or "r=" in plot:
                    ax1.plot(
                        xaxis,
                        h_ratio,
                        color=color,
                        marker=linestyle,
                        ms=10,
                        lw=0,
                        markeredgewidth=4,
                    )
                else:
                    if len(binning) > 20:  # draw ratio as line
                        ax1.plot(xaxis, h_ratio, color=color, linestyle="-", lw=4)
                    else:  # draw as markers
                        ax1.plot(xaxis, h_ratio, color=color, marker="o", ms=10, lw=0)
                sep_power = self._separation_power(dist, reference_hist, binning)
                print("Separation power for hist '%s' is %.4f" % (xlabel, sep_power))

        if ratio:
            self.FormatFig(xlabel="", ylabel=ylabel, ax0=ax0)
            plt.ylabel("Ratio")
            plt.xlabel(xlabel)
            plt.axhline(y=1.0, color="black", linestyle="--", linewidth=1)
            loc = mtick.MultipleLocator(base=10.0)
            ax1.yaxis.set_minor_locator(loc)
            plt.ylim([0.5, 1.5])
        else:
            self.FormatFig(xlabel=xlabel, ylabel=ylabel, ax0=ax0)

        ax0.legend(
            loc=label_loc,
            fontsize=leg_font,
            ncol=1,
            facecolor="white",
            framealpha=0.5,
            frameon=True,
        )
        # plt.tight_layout()
        if ratio:
            plt.subplots_adjust(
                left=0.15, right=0.9, top=0.94, bottom=0.12, wspace=0, hspace=0
            )
        return fig, ax0


class HistERatio(Histogram):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def transform_data(self, data_dict, energies):
        feed_dict = {}
        for key in data_dict:
            dep = np.sum(data_dict[key].reshape(data_dict[key].shape[0], -1), -1)
            if "Geant" in key:
                feed_dict[key] = dep / energies.reshape(-1)
            else:
                feed_dict[key] = dep / energies.reshape(-1)

        # Energy scale is arbitrary, scale so dist centered at 1 for geant
        norm = np.mean(feed_dict[self.geant_key])
        for key in data_dict:
            feed_dict[key] /= norm

        return feed_dict

    def produce_binning(self, reference_data):
        return np.linspace(0.7, 1.3, 30)

    def __call__(self, data_dict, energies):
        feed_dict = self.transform_data(data_dict, energies)

        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Dep. energy / Gen. energy",
            ratio=True,
        )
        for name in self.save_names("ERatio"):
            self.save_fig(name, fig, ax0)


class ScatterESplit(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict, true_energies):
        fig, ax = self.SetFig("Gen. energy [GeV]", "Dep. energy [GeV]")
        for key in data_dict:
            x = true_energies[0:500] if "Geant" in key else true_energies[0:500]
            y = np.sum(data_dict[key].reshape(data_dict[key].shape[0], -1), -1)[0:500]

            ax.scatter(x, y, label=key)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=16, ncol=1)
        plt.tight_layout()
        if len(self.flags.plot_label) > 0:
            ax.set_title(
                self.flags.plot_label, fontsize=20, loc="right", style="italic"
            )
        for name in self.save_names("ScatterES"):
            fig.savefig(name)


class AverageShowerWidth(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:

        def GetMatrix(sizex, minval=-1, maxval=1, binning=None):
            nbins = sizex
            if binning is None:
                binning = np.linspace(minval, maxval, nbins + 1)
            coord = [
                (binning[i] + binning[i + 1]) / 2.0 for i in range(len(binning) - 1)
            ]
            matrix = np.array(coord)
            return matrix

        # TODO : Use radial bins

        phi_matrix = GetMatrix(self.shape_plot[3], minval=-math.pi, maxval=math.pi)
        phi_matrix = np.reshape(phi_matrix, (1, 1, phi_matrix.shape[0]))

        r_matrix = GetMatrix(self.shape_plot[4], minval=0, maxval=self.shape_plot[4])
        r_matrix = np.reshape(r_matrix, (1, 1, r_matrix.shape[0]))

        def GetCenter(matrix, energies, power=1):
            ec = energies * np.power(matrix, power)
            layerE = np.sum(
                np.reshape(energies, (energies.shape[0], energies.shape[1], -1)), -1
            )
            totalE = np.sum(layerE, axis=-1, keepdims=True)
            layer_zero = layerE < (1e-6 * totalE)
            ec = np.reshape(ec, (ec.shape[0], ec.shape[1], -1))  # get value per layer
            ec = np.ma.divide(np.sum(ec, -1), layerE).filled(0)
            ec[layer_zero] = 0.
            return ec

        feed_dict_phi = {}
        feed_dict_phi2 = {}
        feed_dict_r = {}
        feed_dict_r2 = {}

        for key in data_dict:

            data = data_dict[key]

            phi_preprocessed = np.reshape(
                data, (data.shape[0], self.shape_plot[2], self.shape_plot[3], -1)
            )
            phi_proj = preprocessed = np.sum(phi_preprocessed, axis=-1)

            r_preprocessed = np.reshape(
                data, (data.shape[0], self.shape_plot[2], self.shape_plot[4], -1)
            )
            r_proj = preprocessed = np.sum(r_preprocessed, axis=-1)

            feed_dict_phi[key], feed_dict_phi2[key] = ang_center_spread(
                phi_matrix, phi_proj
            )
            feed_dict_r[key] = GetCenter(r_matrix, r_proj)
            feed_dict_r2[key] = GetWidth(
                feed_dict_r[key], GetCenter(r_matrix, r_proj, 2)
            )

        if self.config.get("cartesian_plot", False):
            xlabel1 = "x"
            f_str1 = "Eta"
            xlabel2 = "y"
            f_str2 = "Phi"
        else:
            xlabel1 = "r"
            f_str1 = "R"
            xlabel2 = "alpha"
            f_str2 = "Alpha"
        fig, ax0 = self._plot(
            feed_dict_r,
            xlabel="Layer number",
            ylabel="%s-center of energy" % xlabel1,
        )
        for name in self.save_names(f"FCC{f_str1}EC"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            feed_dict_phi,
            xlabel="Layer number",
            ylabel="%s-center of energy" % xlabel2,
        )
        for name in self.save_names(f"FCC{f_str2}EC"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            feed_dict_r2,
            xlabel="Layer number",
            ylabel="%s-width" % xlabel1,
        )
        for name in self.save_names(f"{f_str1}W"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            feed_dict_phi2,
            xlabel="Layer number",
            ylabel="%s-width (radians)" % xlabel2,
        )
        for name in self.save_names(f"{f_str2}W"):
            self.save_fig(name, fig, ax0)


class ELayer(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            preprocessed = np.reshape(data, (data.shape[0], self.shape_plot[2], -1))
            layer_sum = np.sum(preprocessed, axis=-1)
            totalE = np.sum(preprocessed,axis=(1,2)).reshape(-1,1)
            layer_mean = np.mean(layer_sum, 0)
            layer_std = np.std(layer_sum, 0) / layer_mean
            layer_nonzero = layer_sum > (1e-6 * totalE)
            # preprocessed = np.mean(preprocessed,0)
            return layer_mean, layer_std, layer_nonzero

        feed_dict_avg = {}
        feed_dict_std = {}
        feed_dict_nonzero = {}
        for key in data_dict:
            feed_dict_avg[key], feed_dict_std[key], feed_dict_nonzero[key] = (
                _preprocess(data_dict[key])
            )

        fig, ax0 = self._plot(
            feed_dict_avg,
            xlabel="Layer number",
            ylabel="Mean dep. energy [GeV]",
            no_mean=True,
        )
        for name in self.save_names("EnergyZ"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            feed_dict_std,
            xlabel="Layer number",
            ylabel="Std. Dev. / Mean of energy [GeV]",
            no_mean=True,
        )
        for name in self.save_names("StdEnergyZ"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            feed_dict_nonzero,
            xlabel="Layer number",
            ylabel="Freq. > $10^{-6}$ Total Energy",
        )
        for name in self.save_names("NonZeroEnergyZ"):
            self.save_fig(name, fig, ax0)


class AverageER(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:

        def _preprocess(data):
            preprocessed = np.transpose(data, (0, 4, 1, 2, 3))
            preprocessed = np.reshape(
                preprocessed, (data.shape[0], self.shape_plot[4], -1)
            )
            preprocessed = np.sum(preprocessed, -1)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        if self.config.get("cartesian_plot", False):
            xlabel = "x-bin"
            f_str = "X"
        else:
            xlabel = "R-bin"
            f_str = "R"

        fig, ax0 = self._plot(
            feed_dict,
            xlabel=xlabel,
            ylabel="Mean Energy [GeV]",
        )
        for name in self.save_names(f"Energy_{f_str}"):
            self.save_fig(name, fig, ax0)


class AverageEPhi(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            preprocessed = np.transpose(data, (0, 3, 1, 2, 4))
            preprocessed = np.reshape(
                preprocessed, (data.shape[0], self.shape_plot[3], -1)
            )
            preprocessed = np.sum(preprocessed, -1)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        if self.config.get("cartesian_plot", False):
            xlabel = "y-bin"
            f_str = "Y"
        else:
            xlabel = "alpha-bin"
            f_str = "Alpha"

        fig, ax0 = self._plot(
            feed_dict,
            xlabel=xlabel,
            ylabel="Mean Energy [GeV]",
        )
        for name in self.save_names(f"Energy{f_str}"):
            self.save_fig(name, fig, ax0)


class SparsityLayer(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            eps = 1e-6
            preprocessed = np.reshape(data, (data.shape[0], self.shape_plot[2], -1))
            layer_sparsity = np.sum(preprocessed > eps, axis=-1) / preprocessed.shape[2]
            mean_sparsity = np.mean(layer_sparsity, axis=0)
            std_sparsity = np.std(layer_sparsity, axis=0)
            # preprocessed = np.mean(preprocessed,0)
            return mean_sparsity, std_sparsity

        feed_dict_avg = {}
        feed_dict_std = {}
        feed_dict_nonzero = {}
        for key in data_dict:
            feed_dict_avg[key], feed_dict_std[key] = _preprocess(data_dict[key])

        fig, ax0 = self._plot(
            feed_dict_avg, xlabel="Layer number", ylabel="Mean sparsity", no_mean=True
        )
        for name in self.save_names(f"SparsityZ"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            feed_dict_std,
            xlabel="Layer number",
            ylabel="Std. dev. sparsity",
            no_mean=True,
        )
        for name in self.save_names(f"StdSparsityZ"):
            self.save_fig(name, fig, ax0)


class RadialEnergyHGCal(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:

        geom_file = self.config.get("BIN_FILE", "")
        geom = HGCal_utils.load_geom(geom_file)
        r_vals = geom.ring_map[:, : geom.max_ncell]

        feed_dict = {}
        for key in data_dict:
            nrings = int(np.max(geom.nrings))
            r_bins = np.zeros((data_dict[key].shape[0], nrings))
            for i in range(nrings):
                mask = r_vals == i
                r_bins[:, i] = np.sum(data_dict[key] * mask, axis=(1, 2))

            feed_dict[key] = r_bins

        fig, ax0 = self._plot(feed_dict, xlabel="R-bin", ylabel="Avg. Energy")

        for name in self.save_names(f"EnergyR"):
            self.save_fig(name, fig, ax0)

        return feed_dict


class RCenterHGCal(Histogram):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def transform_data(self, data_dict, energies):
        geom_file = self.config.get("BIN_FILE", "")
        geom = HGCal_utils.load_geom(geom_file)
        r_vals_manual = (geom.xmap[:, : geom.max_ncell]**2 + geom.ymap[:, : geom.max_ncell]**2)**0.5
        r_vals = r_vals_manual

        feed_dict_C_hist = {}
        feed_dict_C_avg = {}
        feed_dict_W_hist = {}
        feed_dict_W_avg = {}
        for key in data_dict:
            # center
            data = data_dict[key]
            #filter out the negligligible layers
            preprocessed = np.reshape(data, (data.shape[0], self.shape_plot[2], -1))
            layer_sum = np.sum(preprocessed, axis=-1)
            totalE = np.sum(preprocessed,axis=(1,2)).reshape(-1,1)
            layer_zero = layer_sum < (1e-6 * totalE)


            r_centers = WeightedMean(r_vals, np.squeeze(data_dict[key]))
            r2_centers = WeightedMean(r_vals, np.squeeze(data_dict[key]), power=2)
            r_centers[layer_zero] = 0.
            r2_centers[layer_zero] = 0.

            feed_dict_C_hist[key] = np.reshape(r_centers, (-1))
            feed_dict_C_avg[key] = np.mean(r_centers, axis=0)

            # width
            r_widths = GetWidth(r_centers, r2_centers)
            feed_dict_W_hist[key] = np.reshape(r_widths, (-1))
            feed_dict_W_avg[key] = np.mean(r_widths, axis=0)

        return {
            "C_hist": feed_dict_C_hist,
            "C_avg": feed_dict_C_avg, 
            "W_hist": feed_dict_W_hist, 
            "W_avg": feed_dict_W_avg
        }

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        processed = self.transform_data(data_dict, energies)

        fig, ax0 = self._hist(
            processed['C_hist'], xlabel="Shower R Center", normalize=True
        )
        for name in self.save_names("RCenter"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            processed['C_avg'], ylabel="Avg. Shower R Center", xlabel="Layer", no_mean=True
        )
        for name in self.save_names("RCenterLayer"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._hist(processed['W_hist'], xlabel="Shower R Width", normalize=True)
        for name in self.save_names("RWidth"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            processed['W_avg'], ylabel="Avg. Shower R Width", xlabel="Layer", no_mean=True
        )
        for name in self.save_names("RWidthLayer"):
            self.save_fig(name, fig, ax0)

        return


class PhiCenterHGCal(Histogram):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def transform_data(self, data_dict, *args, **kwargs):
        
        geom_file = self.config.get("BIN_FILE", "")
        geom = HGCal_utils.load_geom(geom_file)
        phi_vals = geom.theta_map[:, : geom.max_ncell]

        feed_dict_C_hist = {}
        feed_dict_C_avg = {}
        feed_dict_W_hist = {}
        feed_dict_W_avg = {}
        for key in data_dict:
            # center
            data = data_dict[key]
            preprocessed = np.reshape(data, (data.shape[0], self.shape_plot[2], -1))
            layer_sum = np.sum(preprocessed, axis=-1)
            totalE = np.sum(preprocessed,axis=(1,2)).reshape(-1,1)
            layer_zero = layer_sum < (1e-6 * totalE)

            phi_centers, phi_widths = ang_center_spread(
                phi_vals, np.squeeze(data_dict[key])
            )

            phi_centers[layer_zero] = 0.
            phi_widths[layer_zero] = 0.
            feed_dict_C_hist[key] = np.reshape(phi_centers, (-1))
            feed_dict_C_avg[key] = np.mean(phi_centers, axis=0)

            # width
            feed_dict_W_hist[key] = np.reshape(phi_widths, (-1))
            feed_dict_W_avg[key] = np.mean(phi_widths, axis=0)

        return {
            "C_hist": feed_dict_C_hist,
            "C_avg": feed_dict_C_avg, 
            "W_hist": feed_dict_W_hist, 
            "W_avg": feed_dict_W_avg
        }

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        processed = self.transform_data(data_dict)

        fig, ax0 = self._hist(
            processed['C_hist'],
            xlabel="Shower Phi Center",
            ylabel="Arbitrary units",
            normalize=True,
        )
        for name in self.save_names(f"PhiCenter"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            processed['C_avg'],
            ylabel="Avg. Shower Phi Center",
            xlabel="Layer",
            no_mean=True,
        )
        for name in self.save_names(f"PhiCenterLayer"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._hist(
            processed['W_hist'],
            xlabel="Shower Phi Width",
            ylabel="Arbitrary units",
            normalize=True,
        )
        for name in self.save_names(f"PhiWidth"):
            self.save_fig(name, fig, ax0)

        fig, ax0 = self._plot(
            processed['W_avg'],
            ylabel="Avg. Shower Phi Width",
            xlabel="Layer",
            no_mean=True,
        )
        for name in self.save_names(f"PhiWidthLayer"):
            self.save_fig(name, fig, ax0)

        return


class HistEtot(Histogram):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def produce_binning(self, reference_data):
        binning = np.geomspace(
            np.quantile(reference_data[reference_data > 0.], 0.01),
            np.quantile(reference_data, 1.0),
            20,
        )
        return binning

    def transform_data(self, data_dict, *args, **kwargs):
        def _preprocess(data):
            preprocessed = np.reshape(data, (data.shape[0], -1))
            return np.sum(preprocessed, -1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])
        return feed_dict
    
    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        feed_dict = self.transform_data(data_dict)

        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Deposited energy [GeV]",
            reference_name=self.geant_key
        )

        ax0.set_xscale("log")
        for name in self.save_names("TotalE"):
            self.save_fig(name, fig, ax0)


class HistNhits(Histogram):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def produce_binning(self, reference_data):
        vMax = np.max(reference_data) # TODO method to ensure vMax is not in not-reference
        return np.linspace(np.min(reference_data), vMax, 20)
    
    @staticmethod
    def _preprocess(data): 
        min_voxel = 1e-3  # 1 Mev
        preprocessed = np.reshape(data, (data.shape[0], -1))
        return np.sum(preprocessed > min_voxel, -1)

    def transform_data(self, data_dict, *args, **kwargs):
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = HistNhits._preprocess(data_dict[key])
        return feed_dict

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        feed_dict = self.transform_data(data_dict)

        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Number of hits (> 1 MeV)",
            label_loc="upper right",
            reference_name=self.geant_key,
            ratio=True,
        )

        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0, 0))
        ax0.yaxis.set_major_formatter(yScalarFormatter)
        for name in self.save_names("Nhits"):
            self.save_fig(name, fig, ax0)


class HistVoxelE(Histogram):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def produce_binning(self, reference_data):
        vMax = np.max(reference_data)
        vMin = np.amin(reference_data[reference_data > 0])
        binning = np.geomspace(vMin, vMax, 50)
        return binning

    @staticmethod
    def _preprocess(data, nShowers): 
        nShowers = min(nShowers, data.shape[0])
        return np.reshape(data[:nShowers], (-1))
        
    def transform_data(self, data_dict, *args, **kwargs):
        nShowers = 1000
        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = HistVoxelE._preprocess(data_dict[key], nShowers)

        return feed_dict
    
    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:

        feed_dict = self.transform_data(data_dict)

        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Voxel Energy [GeV]",
            reference_name=self.geant_key,
            ratio=True,
            normalize=False,
        )

        ax0.set_xscale("log")
        for name in self.save_names("VoxelE"):
            self.save_fig(name, fig, ax0)


class HistMaxELayer(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            preprocessed = np.reshape(data, (data.shape[0], self.shape_plot[2], -1))
            preprocessed = np.ma.divide(
                np.max(preprocessed, -1), np.sum(preprocessed, -1)
            ).filled(0)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        fig, ax0 = self._plot(
            feed_dict,
            xlabel="Layer number",
            ylabel="Max voxel/Dep. energy",
        )
        for name in self.save_names("MaxEnergyZ"):
            self.save_fig(name, fig, ax0)


class HistMaxE(Histogram):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def produce_binning(self, reference_data):
        return np.linspace(0, 1, 10)
    
    def transform_data(self, data_dict, *args, **kwargs):
        def _preprocess(data):
            preprocessed = np.reshape(data, (data.shape[0], -1))
            preprocessed = np.ma.divide(
                np.max(preprocessed, -1), np.sum(preprocessed, -1)
            ).filled(0)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        return feed_dict
    
    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        feed_dict = self.transform_data(data_dict)
        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Max. voxel/Dep. energy",
        )
        for name in self.save_names("MaxEnergy"):
            self.save_fig(name, fig, ax0)


class Plot_Shower_2D(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)
        plt.rcParams["pcolor.shading"] = "nearest"
        self.layer_number = [10, 44]

    def plot_shower(self, shower, fout="", title="", vmax=0, vmin=0):
        # cmap = plt.get_cmap('PiYG')
        cmap = copy.copy(plt.get_cmap("viridis"))
        cmap.set_bad("white")

        shower[shower == 0] = np.nan

        fig, ax = self.SetFig("x-bin", "y-bin")
        if vmax == 0:
            vmax = np.nanmax(shower[:, :, 0])
            vmin = np.nanmin(shower[:, :, 0])
        im = ax.pcolormesh(
            range(shower.shape[0]),
            range(shower.shape[1]),
            shower[:, :, 0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0, 0))
        # cbar.ax.set_major_formatter(yScalarFormatter)

        fig.colorbar(im, ax=ax, label="Dep. energy [GeV]", format=yScalarFormatter)
        ax.set_title(title, fontsize=15)

        if len(fout) > 0:
            fig.savefig(fout)
        return vmax, vmin

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        for layer in self.layer_number:

            def _preprocess(data):
                preprocessed = data[:, layer, :]
                preprocessed = np.mean(preprocessed, 0)
                preprocessed[preprocessed == 0] = np.nan
                return preprocessed

            vmin = vmax = 0
            nShowers = 5
            for ik, key in enumerate(
                ["Geant4", utils.name_translate[self.flags.model]]
            ):
                average = _preprocess(data_dict[key])

                fout_avg = self.save_names(f"{key}2D_{layer}")[0]
                title = "{}, layer number {}".format(key, layer)
                self.plot_shower(average, fout=fout_avg, title=title)

                for i in range(nShowers):
                    shower = data_dict[key][i, layer]
                    fout_ex = self.save_names(f"{key}2D_{layer}_shower{i}")[0]

                    title = "{} Shower {}, layer number {}".format(key, i, layer)
                    vmax, vmin = self.plot_shower(
                        shower, fout=fout_ex, title=title, vmax=vmax, vmin=vmin
                    )



def plot_shower_layer(data, fname = "", title=None, fig=None, subplot=(1, 1, 1),
                     vmin = None, vmax=None, colbar='alone', r_edges = None):
    """ draws the shower in layer_nr only """
    if fig is None:
        fig = plt.figure(figsize=(5, 5), dpi=200)

    #r,phi
    shape = data.shape
    nPhi = shape[0]
    nRad = shape[1]

    pts_per_angular_bin = 50
    num_splits = pts_per_angular_bin * nPhi


    if(r_edges is None):
        r_edges = np.arange(nRad + 1) 
    max_r = np.amax(r_edges)

    phi_bins = 2.*np.pi* np.arange(num_splits+1)/ num_splits

    theta, rad = np.meshgrid(phi_bins, r_edges)
    data_reshaped = data.reshape(nPhi, -1)
    data_repeated = np.repeat(data_reshaped, (pts_per_angular_bin), axis=0)

    ax = fig.add_subplot(*subplot, polar=True)
    #ax = plt.subplot(111, projection='polar')
    ax.grid(False)
    if vmax is None: vmax = data.max()
    if vmin is None: vmin = 1e-2 if data.max() > 1e-3 else data.max()/100.


    pcm = ax.pcolormesh(theta, rad, data_repeated.T+1e-16, norm=LN(vmin=vmin, vmax=vmax))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_rmax(max_r)

    if title is not None:
        ax.set_title(title)

    if colbar == 'alone':
        axins = inset_axes(fig.get_axes()[-1], width='100%',
                           height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                           bbox_transform=fig.get_axes()[-1].transAxes,
                           borderpad=0)
        cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
        cbar.set_label(r'Energy (GeV)', y=0.83, fontsize=12)
    elif colbar == 'both':
        axins = inset_axes(fig.get_axes()[-1], width='200%',
                           height="15%", loc='lower center',
                           bbox_to_anchor=(-0.625, -0.2, 1, 1),
                           bbox_transform=fig.get_axes()[-1].transAxes,
                           borderpad=0)
        cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
        cbar.set_label(r'Energy (GeV)', y=0.83, fontsize=12)
    elif colbar == 'None':
        pass

    plt.subplots_adjust(bottom=0.25, left = 0.02, right = 0.98, top = 0.95)


    if fname is not None:
        plt.savefig(fname, facecolor='white')
