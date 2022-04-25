import matplotlib.pyplot as plt
import numpy as num
from lightguide.rust import goldstein_filter_rect
from lightguide.utils import traces_to_numpy_and_meta
from matplotlib import patches
from pyrocko import io
from scipy import signal

deltat = 0.001
sampling_rate = 1.0 / deltat
data, meta = traces_to_numpy_and_meta(
    io.load("data/landwuest_UTC_20210422_034648.386.tdms", format="detect")
)

time_lim = (13.8, 18.0)
chan_lim = (660, 260)
panel = "abcdefgh"

imshow_args = dict(
    aspect="auto",
    extent=(0, data.shape[1] * deltat, data.shape[0], 0),
    interpolation="none",
    cmap="binary",
)

lowpass_filter = signal.butter(
    4, 200.0, btype="lowpass", fs=sampling_rate, output="sos"
)
data = signal.sosfilt(lowpass_filter, data, axis=1)
data = data.astype(num.float32)


def plot_data(data, ax, normalize=True):
    if normalize:
        data = data / num.abs(data).max(axis=1)[:, num.newaxis]
    ax.imshow(data, **imshow_args)
    ax.set_ylabel("Channel #")
    ax.set_xlabel("Time [s]")
    ax.set_xlim(*time_lim)
    ax.set_ylim(*chan_lim)


filter_sizes = ((16, 0.05), (16, 0.1), (32, 0.2), (32, 0.4))


fig = plt.figure(figsize=(10, 8))
axes = fig.subplots(len(filter_sizes) + 1, 1, sharex="all", sharey="all")

ax = axes[0]
plot_data(data, ax)
ax.text(time_lim[0] + 0.05, chan_lim[0] - 60, "Input data")


for ifilt, (filt_ntraces, filt_length) in enumerate(filter_sizes):
    ax = axes[ifilt + 1]
    data = data.copy()

    filt_x = filt_ntraces
    filt_y = int(round(filt_length / deltat))

    overlap_x = int(filt_x / 2 - 1)
    overlap_y = int(filt_y / 2 - 1)

    data = goldstein_filter_rect(
        data,
        window_size=(filt_x, filt_y),
        overlap=(overlap_x, overlap_y),
        exponent=1.0,
        normalize_power=True,
    )

    plot_data(data, ax)

    ax.add_artist(
        patches.Rectangle(
            (time_lim[0] + 0.05, chan_lim[0] - 60),
            filt_length,
            filt_ntraces,
            fc="w",
            lw=0,
            alpha=0.5,
        )
    )


for iax, ax in enumerate(axes):
    ax.text(
        0.01,
        0.95,
        f"{panel[iax]})",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    ax.axvline(14.05, c="w", ls="--", lw="1.0")
    if ax != axes[-1]:
        ax.set_xlabel(None)

fig.tight_layout()
fig.savefig("figures/rectangular-comparison.png")

plt.show()
