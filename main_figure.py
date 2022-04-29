from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import matplotlib
from matplotlib import patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from lightguide.rust import goldstein_filter_rect as goldstein_filter_rust
from lightguide.utils import traces_to_numpy_and_meta
from pyrocko import io

matplotlib.use("Qt5Agg")


@dataclass
class FilterOptions:
    exponent: float
    default: bool = False
    linestyle: str = "--"
    alpha: float = 0.8
    normalize_power: bool = False
    color: str = "black"
    window_size: int | tuple[int, int] = 16
    _trace: Optional[plt.Line2D] = None
    _spec: Optional[plt.Line2D] = None

    @property
    def overlap(self) -> tuple[int, int]:
        if isinstance(self.window_size, int):
            window_size = (self.window_size, self.window_size)
        else:
            window_size = self.window_size
        return (int(window_size[0] / 2 - 1), int(window_size[1] / 2 - 1))


def plot_goldstein_analysis(
    data,
    filters: list[FilterOptions],
    trace_select: int = 500,
    sampling_rate: float = 200.0,
    time_extent: tuple = (14, 20),
    channel_extent: tuple = (330, 550),
    noise_band: tuple[float, float] = (20.0, 50.0),
    signal_window: tuple[float, float] = (0.0, 10.0),
    save_filename: Optional[str] = None,
    show_plot: bool = True,
):
    def r(data):
        v = np.std(data) * 2
        return -v, v

    def normalize_diff(data, filtered_data):
        data_max = np.abs(data).max()
        filtered_data = filtered_data / np.abs(filtered_data).max() * data_max
        return filtered_data - data
        # return Normalize()(data) - Normalize()(filtered_data)

    for filter in filters:
        if filter.default:
            default_filter = filter
            break
    else:
        default_filter = filters[0]

    for filter in filters:
        if isinstance(filter.window_size, int):
            filter.window_size = (filter.window_size, filter.window_size)

    maxval = np.abs(data).max()
    imshow_kwargs = dict(
        aspect="auto",
        cmap="binary",
        interpolation="none",
        vmin=-maxval,
        vmax=maxval,
        extent=(0.0, data.shape[1] / sampling_rate, data.shape[0], 0),
    )

    # Setup Figure
    fig = plt.figure(figsize=(10, 8), tight_layout=True)
    gs = gridspec.GridSpec(3, 3, height_ratios=(2, 1, 1))
    ax_data = fig.add_subplot(gs[0, 0])
    ax_filt = fig.add_subplot(gs[0, 1], sharex=ax_data, sharey=ax_data)
    ax_resi = fig.add_subplot(gs[0, 2], sharex=ax_data, sharey=ax_data)

    ax_trace = fig.add_subplot(gs[1, :], sharex=ax_data)
    ax_spec = fig.add_subplot(gs[2, :], yscale="log", xscale="log")

    ax_data.set_xlim(*time_extent)
    ax_data.set_ylim(*channel_extent)

    data_filtered = goldstein_filter_rust(
        data,
        window_size=default_filter.window_size,
        exponent=min(1.0, default_filter.exponent),
        overlap=default_filter.overlap,
        normalize_power=default_filter.normalize_power,
    ).copy()
    data_filtered[np.isnan(data_filtered)] = 0.0
    data_filtered[np.isinf(data_filtered)] = 0.0
    if not default_filter.normalize_power:
        data_filtered /= np.abs(data_filtered).max() * maxval

    trace_raw = data[trace_select]
    trace_raw -= trace_raw.mean()

    ax_data.imshow(data, **imshow_kwargs)
    ax_filt.imshow(data_filtered, **imshow_kwargs)
    ax_resi.imshow(normalize_diff(data, data_filtered), **imshow_kwargs)

    ax_data.invert_yaxis()
    ax_filt.invert_yaxis()
    ax_resi.invert_yaxis()
    ax_data.axhline(trace_select, ls="--", c="k", alpha=0.5)
    ax_filt.axhline(trace_select, ls="--", c=default_filter.color, alpha=0.5)

    # Hide dublicate labels
    [label.set_visible(False) for label in ax_filt.get_yticklabels()]
    [label.set_visible(False) for label in ax_resi.get_yticklabels()]

    ax_data.set_ylabel("Channel #")
    for ax in (ax_data, ax_filt, ax_resi):
        ax.set_xlabel("Time [s]")

    duration = time_extent[1] - time_extent[0]
    nchannels = channel_extent[1] - channel_extent[0]
    time = np.arange(data.shape[1]) / sampling_rate
    signal_idx = np.logical_and(time > signal_window[0], time < signal_window[1])
    signal_duration = signal_window[1] - signal_window[0]

    # Trace axes
    ax_trace.grid(alpha=0.3)
    ax_trace.spines["top"].set_visible(False)
    ax_trace.spines["right"].set_visible(False)
    ax_trace.set_xlabel("Time [s]")
    ax_trace.set_ylabel(
        ("Norm. " if not default_filter.normalize_power else "") + "Amplitude"
    )
    ax_trace.plot(time, trace_raw, alpha=0.5, c="k", lw=1.0)[0]
    ax_trace.axvspan(*signal_window, fc="#E4E4E4")

    # Spectrum axes
    ax_spec.grid(alpha=0.3)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    ax_spec.set_xlabel("Frequency [Hz]")
    ax_spec.set_ylabel(
        ("Norm. " if not default_filter.normalize_power else "") + "Power"
    )
    ax_spec.set_xlim(2.0 / signal_duration, sampling_rate / 2)
    ax_spec.axvspan(*noise_band, fc="#BCEBCE")

    spec_raw = np.abs(np.fft.rfft(trace_raw[signal_idx])) ** 2
    spec_freq = np.fft.rfftfreq(trace_raw[signal_idx].size, 1.0 / sampling_rate)

    noise_idx = np.logical_and(noise_band[0] < spec_freq, spec_freq < noise_band[1])
    noise_level_raw = np.mean(spec_raw[noise_idx])

    ax_spec.plot(spec_freq, spec_raw, alpha=0.5, c="k", lw=1.0)[0]

    # ax_data.set_title("Data Input")
    # ax_filt.set_title("Data Filtered")
    # ax_resi.set_title("Data Residual")

    for ax, panel in zip((ax_data, ax_filt, ax_resi), "abc"):
        ax.text(0.03, 0.97, f"{panel})", va="top", transform=ax.transAxes)

    ax_filt.text(
        0.05,
        0.05,
        f"$\\alpha = {default_filter.exponent}$"
        + (", normalized" if default_filter.normalize_power else ""),
        transform=ax_filt.transAxes,
        fontsize="small",
        backgroundcolor=(1.0, 1.0, 1.0, 0.3),
    )

    ax_filt.add_artist(
        patches.Rectangle(
            (0.03, 0.8),
            (default_filter.window_size[1] * (1.0 / sampling_rate)) / duration,
            default_filter.window_size[0] / nchannels,
            fc="white",
            lw=0,
            alpha=0.5,
            transform=ax_filt.transAxes,
        )
    )

    ax_trace.text(0.01, 1.0, "d)", va="top", transform=ax_trace.transAxes)
    ax_spec.text(0.01, 1.0, "e)", va="top", transform=ax_spec.transAxes)

    for filter in filters:
        data_filtered = goldstein_filter_rust(
            data,
            window_size=filter.window_size,
            exponent=min(1.0, filter.exponent),
            overlap=filter.overlap,
            normalize_power=filter.normalize_power,
        ).copy()
        data_filtered[np.isnan(data_filtered)] = 0.0
        data_filtered[np.isinf(data_filtered)] = 0.0
        if not filter.normalize_power:
            data_filtered /= np.abs(data_filtered).max() * maxval
        trace_filtered = data_filtered[trace_select]
        trace_filtered_spec = np.abs(np.fft.rfft(trace_filtered[signal_idx])) ** 2

        noise_level_filtered = np.mean(trace_filtered_spec[noise_idx])
        noise_level_db = 10 * np.log10(noise_level_filtered / noise_level_raw)

        ax_trace.plot(
            time,
            trace_filtered,
            alpha=filter.alpha,
            c=filter.color,
            lw=1.0,
        )
        ax_spec.plot(
            spec_freq,
            trace_filtered_spec,
            alpha=filter.alpha,
            c=filter.color,
            lw=1.0,
            label=f"$\\alpha={filter.exponent}$ ({noise_level_db:.1f} dB)",
        )

    ax_spec.legend(loc="lower left", framealpha=0.7)

    if save_filename:
        fig.savefig(save_filename, dpi=150)
    if show_plot:
        plt.show()

    fig.clear()


show_plot = False

if True:
    # VSP Shot
    args = dict(
        data=np.load("data/das-data-vsp.npy"),
        sampling_rate=1000.0,
        trace_select=275,
        time_extent=(1.0, 2.0),
        channel_extent=(150, 350),
        noise_band=(150, 500),
        signal_window=(1.42, 1.75),
        show_plot=show_plot,
    )

    plot_goldstein_analysis(
        filters=[
            FilterOptions(
                exponent=0.6, normalize_power=True, color="darkgreen", alpha=0.5
            ),
            FilterOptions(
                exponent=0.8, normalize_power=True, color="mediumblue", default=True
            ),
            FilterOptions(
                exponent=1.0, normalize_power=True, color="firebrick", alpha=0.5
            ),
        ],
        save_filename="figures/vsp-shot-NAFK.png",
        **args,
    )

    plot_goldstein_analysis(
        filters=[
            FilterOptions(exponent=0.6, color="mediumblue", alpha=0.5),
            FilterOptions(exponent=0.8, color="firebrick", default=True),
            FilterOptions(exponent=1.0, color="darkgreen", alpha=0.5),
        ],
        save_filename="figures/vsp-shot-AFK.png",
        **args,
    )


if True:
    # Regional earthquake
    args = dict(
        data=np.load("data/data-DAS-gfz2020wswf.npy"),
        trace_select=500,
        time_extent=(
            14.0,
            20.0,
        ),
        channel_extent=(330, 550),
        noise_band=(20, 200),
        signal_window=(15.5, 19.0),
        show_plot=show_plot,
    )

    plot_goldstein_analysis(
        filters=[
            FilterOptions(exponent=0.6, color="mediumblue", alpha=0.5),
            FilterOptions(exponent=0.8, color="firebrick", default=True),
            FilterOptions(exponent=1.0, color="darkgreen", alpha=0.5),
        ],
        save_filename="figures/gfz2020wswf-AFK.png",
        **args,
    )

    plot_goldstein_analysis(
        filters=[
            FilterOptions(
                exponent=1.0, normalize_power=True, color="firebrick", alpha=0.5
            ),
            FilterOptions(
                exponent=0.6, normalize_power=True, color="darkgreen", alpha=0.5
            ),
            FilterOptions(
                exponent=0.8, normalize_power=True, color="mediumblue", default=True
            ),
        ],
        save_filename="figures/gfz2020wswf-NAFK.png",
        **args,
    )
