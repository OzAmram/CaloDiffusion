from abc import abstractmethod, ABC
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick

from calodiffusion.utils import utils


class ScalarFormatterClass(mtick.ScalarFormatter):
    # https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"

class Plot(ABC): 
    def __init__(self, flags, config) -> None:

        self.flags = flags
        self.config = config

        self.plt_exts = ["png", "pdf"]


        self.line_style = {
            "Geant4": "dotted",
            "Diffusion": "-",
            "Avg Shower": "-",
            "CaloDiffusion 400 Steps": "-",
            "CaloDiffusion 200 Steps": "-",
            "CaloDiffusion 100 Steps": "-",
            "CaloDiffusion 50 Steps": "-",
        }

        self.colors = {
            "Geant4": "black",
            "Avg Shower": "blue",
            "Diffusion": "blue",
            "CaloDiffusion 400 Steps": "blue",
            "CaloDiffusion 200 Steps": "green",
            "CaloDiffusion 100 Steps": "purple",
            "CaloDiffusion 50 Steps": "red",
        }
        Plot.set_style()


    def save_names(self, plot_name) -> list[str]: 
        return [
            f"{self.flags.plot_folder}/{plot_name}_{self.config['CHECKPOINT_NAME']}_{self.flags.model}.{extension}"
            for extension 
            in self.plt_exts
        ]
    

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

        # #
        mpl.rcParams.update({"font.size": 26})
        # mpl.rcParams.update({'legend.fontsize': 18})
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


    def _hist(self, 
        feed_dict,
            xlabel="",
            ylabel="Arbitrary units",
            reference_name="Geant4",
            logy=False,
            binning=None,
            label_loc="best",
            ratio=True,
            normalize=True,
            leg_font=24,
        ):
            assert (
                reference_name in feed_dict.keys()
            ), "ERROR: Don't know the reference distribution"

            fig, gs = self.SetGrid(ratio)
            ax0 = plt.subplot(gs[0])
            if ratio:
                plt.xticks(fontsize=0)
                ax1 = plt.subplot(gs[1], sharex=ax0)

            if binning is None:
                binning = np.linspace(
                    np.quantile(feed_dict[reference_name], 0.0),
                    np.quantile(feed_dict[reference_name], 1),
                    10,
                )
            xaxis = [(binning[i] + binning[i + 1]) / 2.0 for i in range(len(binning) - 1)]
            reference_hist, _ = np.histogram(
                feed_dict[reference_name], bins=binning, density=True
            )

            for ip, plot in enumerate(reversed(list(feed_dict.keys()))):
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
                        linestyle=self.line_style[plot],
                        color=self.colors[plot],
                        density=True,
                        histtype="step",
                        lw=4,
                    )

                if len(self.flags.plot_label) > 0:
                    ax0.set_title(self.flags.plot_label, fontsize=20, loc="right", style="italic")

                if reference_name != plot and ratio:
                    eps = 1e-8
                    h_ratio = 100 * np.divide(dist - reference_hist, reference_hist + eps)
                    if "steps" in plot or "r=" in plot:
                        ax1.plot(
                            xaxis,
                            h_ratio,
                            color=self.colors[plot],
                            marker=self.line_style[plot],
                            ms=10,
                            lw=0,
                            markeredgewidth=4,
                        )
                    else:
                        if len(binning) > 20:  # draw ratio as line
                            ax1.plot(xaxis, h_ratio, color=self.colors[plot], linestyle="-", lw=4)
                        else:  # draw as markers
                            ax1.plot(
                                xaxis, h_ratio, color=self.colors[plot], marker="o", ms=10, lw=0
                            )
                    sep_power = self._separation_power(dist, reference_hist, binning)
                    print("Separation power for hist '%s' is %.4f" % (xlabel, sep_power))

            if logy:
                ax0.set_yscale("log")

            if ratio:
                self.FormatFig(xlabel="", ylabel=ylabel, ax0=ax0)
                plt.ylabel("Diff. (%)")
                plt.xlabel(xlabel)
                plt.axhline(y=0.0, color="black", linestyle="-", linewidth=1)
                plt.axhline(y=10, color="gray", linestyle="--", linewidth=1)
                plt.axhline(y=-10, color="gray", linestyle="--", linewidth=1)
                loc = mtick.MultipleLocator(base=10.0)
                ax1.yaxis.set_minor_locator(loc)
                plt.ylim([-50, 50])
            else:
                self.FormatFig(xlabel=xlabel, ylabel=ylabel, ax0=ax0)

            ax0.legend(loc=label_loc, fontsize=leg_font, ncol=1)
            # plt.tight_layout()
            if ratio:
                plt.subplots_adjust(
                    left=0.15, right=0.9, top=0.94, bottom=0.12, wspace=0, hspace=0
                )
            return fig, ax0

    def _plot(
        self,
        feed_dict,
        xlabel="",
        ylabel="",
        reference_name="Geant4",
        no_mean=False,
    ):
        assert (
            reference_name in feed_dict.keys()
        ), "ERROR: Don't know the reference distribution"

        fig, gs = self.SetGrid()
        ax0 = plt.subplot(gs[0])
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1], sharex=ax0)

        for ip, plot in enumerate(feed_dict.keys()):
            if no_mean:
                d = feed_dict[plot]
                ref = feed_dict[reference_name]
            else:
                d = np.mean(feed_dict[plot], 0)
                ref = np.mean(feed_dict[reference_name], 0)
            if "steps" in plot or "r=" in plot:
                ax0.plot(d, label=plot, marker=self.line_style[plot], color=self.colors[plot], lw=0)
            else:
                ax0.plot(d, label=plot, linestyle=self.line_style[plot], color=self.colors[plot])
            if len(self.flags.plot_label) > 0:
                ax0.set_title(self.flags.plot_label, fontsize=20, loc="right", style="italic")
            if reference_name != plot:
                ax0.get_xaxis().set_visible(False)
                ax0.set_ymargin(0)

                eps = 1e-8
                ratio = 100 * np.divide(ref - d, d + eps)
                # ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)

                plt.axhline(y=0.0, color="black", linestyle="-", linewidth=2)
                plt.axhline(y=10, color="gray", linestyle="--", linewidth=2)
                plt.axhline(y=-10, color="gray", linestyle="--", linewidth=2)

                if "steps" in plot or "r=" in plot:
                    ax1.plot(
                        ratio,
                        color=self.colors[plot],
                        markeredgewidth=4,
                        marker=self.line_style[plot],
                        lw=0,
                    )
                else:
                    ax1.plot(ratio, color=self.colors[plot], linestyle=self.line_style[plot])

        self.FormatFig(xlabel="", ylabel=ylabel, ax0=ax0)
        ax0.legend(loc="best", fontsize=24, ncol=1)

        plt.ylabel("Diff. (%)")
        plt.xlabel(xlabel)
        loc = mtick.MultipleLocator(base=10.0)
        ax1.yaxis.set_minor_locator(loc)
        plt.ylim([-50, 50])

        plt.subplots_adjust(left=0.15, right=0.9, top=0.94, bottom=0.12, wspace=0, hspace=0)
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

class HistERatio(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict, energies): 
        feed_dict = {}
        for key in data_dict:
            dep = np.sum(data_dict[key].reshape(data_dict[key].shape[0], -1), -1)
            if "Geant" in key:
                feed_dict[key] = dep / energies.reshape(-1)
            else:
                feed_dict[key] = dep / energies.reshape(-1)

        binning = np.linspace(0.5, 1.5, 51)

        fig, _ = self._hist(
            feed_dict,
            xlabel="Dep. energy / Gen. energy",
            logy=False,
            binning=binning,
            ratio=True,
        )
        for name in self.save_names("FCC_ERatio"): 
            fig.savefig(name)

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
            ax.set_title(self.flags.plot_label, fontsize=20, loc="right", style="italic")
        for name in self.save_names("ScatterES"): 
            fig.savefig(name)
            


class AverageShowerWidth(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def GetMatrix(sizex, sizey, minval=-1, maxval=1, binning=None):
            nbins = sizex
            if binning is None:
                binning = np.linspace(minval, maxval, nbins + 1)
            coord = [(binning[i] + binning[i + 1]) / 2.0 for i in range(len(binning) - 1)]
            matrix = np.repeat(np.expand_dims(coord, -1), sizey, -1)
            return matrix

        # TODO : Use radial bins
        # r_bins = [0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85]

        phi_matrix = GetMatrix(
            self.config["SHAPE"][2],
            self.config["SHAPE"][3],
            minval=-math.pi,
            maxval=math.pi,
        )
        phi_matrix = np.reshape(
            phi_matrix, (1, 1, phi_matrix.shape[0], phi_matrix.shape[1], 1)
        )

        r_matrix = np.transpose(
            GetMatrix(self.config["SHAPE"][3], self.config["SHAPE"][2])
        )
        r_matrix = np.reshape(r_matrix, (1, 1, r_matrix.shape[0], r_matrix.shape[1], 1))

        def GetCenter(matrix, energies, power=1):
            ec = energies * np.power(matrix, power)
            sum_energies = np.sum(
                np.reshape(energies, (energies.shape[0], energies.shape[1], -1)), -1
            )
            ec = np.reshape(ec, (ec.shape[0], ec.shape[1], -1))  # get value per layer
            ec = np.ma.divide(np.sum(ec, -1), sum_energies).filled(0)
            return ec

        def ang_center_spread(matrix, energies):
            # weighted average over periodic variabel (angle)
            # https://github.com/scipy/scipy/blob/v1.11.1/scipy/stats/_morestats.py#L4614
            # https://en.wikipedia.org/wiki/Directional_statistics#The_fundamental_difference_between_linear_and_circular_statistics
            cos_matrix = np.cos(matrix)
            sin_matrix = np.sin(matrix)
            cos_ec = GetCenter(cos_matrix, energies)
            sin_ec = GetCenter(sin_matrix, energies)
            ang_mean = np.arctan2(sin_ec, cos_ec)
            R = sin_ec**2 + cos_ec**2
            eps = 1e-8
            R = np.clip(R, eps, 1.0)

            ang_std = np.sqrt(-np.log(R))
            return ang_mean, ang_std

        def GetWidth(mean, mean2):
            width = np.ma.sqrt(mean2 - mean**2).filled(0)
            return width

        feed_dict_phi = {}
        feed_dict_phi2 = {}
        feed_dict_r = {}
        feed_dict_r2 = {}

        for key in data_dict:
            feed_dict_phi[key], feed_dict_phi2[key] = ang_center_spread(
                phi_matrix, data_dict[key]
            )
            feed_dict_r[key] = GetCenter(r_matrix, data_dict[key])
            feed_dict_r2[key] = GetWidth(
                feed_dict_r[key], GetCenter(r_matrix, data_dict[key], 2)
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
            fig.savefig(name)
            
        fig, ax0 = self._plot(
            feed_dict_phi,
            xlabel="Layer number",
            ylabel="%s-center of energy" % xlabel2,
        )
        for name in self.save_names(f"FCC{f_str2}EC"):
            fig.savefig(name)

        fig, ax0 = self._plot(
            feed_dict_r2,
            xlabel="Layer number",
            ylabel="%s-width" % xlabel1,
        )
        for name in self.save_names(f"FCC_{f_str1}W"):
            fig.savefig(name)

        fig, ax0 = self._plot(
            feed_dict_phi2,
            xlabel="Layer number",
            ylabel="%s-width (radians)" % xlabel2,
        )
        for name in self.save_names(f"FCC_{f_str2}W"):
            fig.savefig(name)

class ELayer(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def _preprocess(self, data, total_events):
        preprocessed = np.reshape(data, (total_events, self.config["SHAPE"][1], -1))
        layer_sum = np.sum(preprocessed, -1)
        totalE = np.sum(preprocessed, -1)
        layer_mean = np.mean(layer_sum, 0)
        layer_std = np.std(layer_sum, 0) / layer_mean
        layer_nonzero = layer_sum > (1e-6 * totalE)
        # preprocessed = np.mean(preprocessed,0)
        return layer_mean, layer_std, layer_nonzero

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        feed_dict_avg = {}
        feed_dict_std = {}
        feed_dict_nonzero = {}
        for key in data_dict:
            feed_dict_avg[key], feed_dict_std[key], feed_dict_nonzero[key] = self._preprocess(
                data_dict[key], total_events=data_dict[key].shape[0]
            )

        fig, ax0 = self._plot(
            feed_dict_avg,
            xlabel="Layer number",
            ylabel="Mean dep. energy [GeV]",
            no_mean=True,
        )
        for name in self.save_names("FCC_EnergyZ"):
            fig.savefig(name)

        fig, ax0 = self._plot(
            feed_dict_std,
            xlabel="Layer number",
            ylabel="Std. Dev. / Mean of energy [GeV]",
            no_mean=True,
        )
        for name in self.save_names("FCC_StdEnergyZ"):
            fig.savefig(name)

        fig, ax0 = self._plot(
            feed_dict_nonzero,
            xlabel="Layer number",
            ylabel="Freq. > $10^{-6}$ Total Energy",
        )
        for name in self.save_names("C_NonZeroEnergyZ"):
            fig.savefig(name)

class AverageER(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
    
        def _preprocess(data):
            preprocessed = np.transpose(data, (0, 3, 1, 2, 4))
            preprocessed = np.reshape(
                preprocessed, (data.shape[0], self.config["SHAPE"][3], -1)
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
        for name in self.save_names(f"FCC_Energy_{f_str}"):
            fig.savefig(name)


class AverageEPhi(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            preprocessed = np.transpose(data, (0, 2, 1, 3, 4))
            preprocessed = np.reshape(
                preprocessed, (data.shape[0], self.config["SHAPE"][2], -1)
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
        for name in self.save_names(f"FCC_Energy{f_str}"):
            fig.savefig(name)


class HistEtot(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            preprocessed = np.reshape(data, (data.shape[0], -1))
            return np.sum(preprocessed, -1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        # binning = np.geomspace(np.quantile(feed_dict['Geant4'],0.01),np.quantile(feed_dict['Geant4'],1.0),20)
        binning = np.geomspace(1.0, np.amax(feed_dict["Geant4"]), 20)
        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Deposited energy [GeV]",
            logy=True,
            binning=binning,
        )
        ax0.set_xscale("log")
        for name in self.save_names("FCC_TotalE"):
            fig.savefig(name)


class HistNhits(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)
    
    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            min_voxel = 1e-3  # 1 Mev
            preprocessed = np.reshape(data, (data.shape[0], -1))
            return np.sum(preprocessed > min_voxel, -1)

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        binning = np.linspace(
            np.quantile(feed_dict["Geant4"], 0.0), np.quantile(feed_dict["Geant4"], 1), 20
        )
        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Number of hits (> 1 MeV)",
            label_loc="upper right",
            binning=binning,
            ratio=True,
        )
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0, 0))
        ax0.yaxis.set_major_formatter(yScalarFormatter)
        for name in self.save_names("FCC_Nhits"):
            fig.savefig(name)


class HistVoxelE(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)

    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
    
        def _preprocess(data):
            return np.reshape(data, (-1))

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        vmin = np.amin(feed_dict["Geant4"][feed_dict["Geant4"] > 0])
        binning = np.geomspace(vmin, np.quantile(feed_dict["Geant4"], 1.0), 50)
        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Voxel Energy [GeV]",
            logy=True,
            binning=binning,
            ratio=True,
            normalize=False,
        )

        ax0.set_xscale("log")
        for name in self.save_names("FCC_VoxelE"):
            fig.savefig(name)


class HistMaxELayer(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)
        
    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
        def _preprocess(data):
            preprocessed = np.reshape(data, (data.shape[0], self.config["SHAPE"][1], -1))
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
        for name in self.save_names("FCC_MaxEnergyZ"):
            fig.savefig(name)


class HistMaxE(Plot):
    def __init__(self, flags, config) -> None:
        super().__init__(flags, config)
    
    def __call__(self, data_dict: dict[str, np.ndarray], energies: np.ndarray) -> None:
    
        def _preprocess(data):
            preprocessed = np.reshape(data, (data.shape[0], -1))
            preprocessed = np.ma.divide(
                np.max(preprocessed, -1), np.sum(preprocessed, -1)
            ).filled(0)
            return preprocessed

        feed_dict = {}
        for key in data_dict:
            feed_dict[key] = _preprocess(data_dict[key])

        binning = np.linspace(0, 1, 10)
        fig, ax0 = self._hist(
            feed_dict,
            xlabel="Max. voxel/Dep. energy",
            binning=binning,
            logy=True,
        )
        for name in self.save_names("FCC_MaxEnergy"):
            fig.savefig(name)


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
            # print(vmin,vmax)
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

                fout_avg = self.save_names(f"FCC_{key}2D_{layer}")[0]
                title = "{}, layer number {}".format(key, layer)
                self.plot_shower(average, fout=fout_avg, title=title)

                for i in range(nShowers):
                    shower = data_dict[key][i, layer]
                    fout_ex = self.save_names(f"FCC_{key}2D_{layer}_shower{i}")[0]

                    title = "{} Shower {}, layer number {}".format(key, i, layer)
                    vmax, vmin = self.plot_shower(
                        shower, fout=fout_ex, title=title, vmax=vmax, vmin=vmin
                    )
