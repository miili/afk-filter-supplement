from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from lightguide.rust import goldstein_filter_rect as goldstein_filter
from matplotlib import gridspec, patches

rand = np.random.RandomState(123)

DELTA_X = 0.01
DELTA_T = 0.01


def get_sweeping_wavefield(
    size: tuple[int, int],
    amplitude: float,
    freq_min: float,
    freq_max: float,
    slowness: float = 0.0,
    dx: float = DELTA_X,
    dt: float = DELTA_T,
    fade_in=2.0,
    kernel_size=500,
):
    coords = np.mgrid[-size[0] / 2 : size[0] / 2, -size[1] / 2 : size[1] / 2]
    coords[0] *= dx
    coords[1] *= dt

    omega_min = 2 * np.pi * freq_min
    omega_max = 2 * np.pi * freq_max

    omega = np.linspace(omega_min, omega_max, num=size[0])

    if slowness:
        coords[1] += coords[0] / slowness

    taper = coords[1].copy()
    taper[taper < fade_in] = 0.0
    taper[taper >= fade_in] = 1.0

    # Smooth the taper
    kernel = np.ones(kernel_size) / kernel_size
    taper = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=taper
    )

    field = np.sin(omega[:, np.newaxis] * coords[1])
    field *= taper
    return amplitude * field.astype(np.float32)


def get_spectral_noise_like(arr: npt.NDArray[np.float32], amplitude: float):
    spec = np.fft.rfft(np.zeros_like(arr), axis=1)
    amplitude *= spec.shape[1] * 2
    spec += rand.normal(size=spec.shape, scale=amplitude)
    return np.fft.irfft(spec, axis=1).astype(np.float32)


def get_norm_spec(data: npt.NDArray):
    spec = np.abs(np.fft.rfft(data, axis=1)) ** 2
    spec = spec.sum(axis=0)
    return spec / spec.max()


def ax_remove_ticks(ax: plt.Axes):
    ax.set_xticks([])
    ax.set_yticks([])


def plot_filter_performance(
    data: npt.NDArray[np.float32],
    noise_amp: float = 1.0,
    noise_band: tuple[float, float] = (5.0, 50.0),
    filename: str = "",
    show: bool = False,
):
    window_sizes = (32, 64, 128, 256)

    fig = plt.figure(figsize=(9, 10), tight_layout=True)
    gs = gridspec.GridSpec(4, len(window_sizes))

    ax_noise = fig.add_subplot(gs[0, :2])
    ax_clean = fig.add_subplot(gs[0, 2:])

    # Add Noise
    noise = get_spectral_noise_like(data, noise_amp)
    data_noisy = data + noise

    imshow_args = dict(cmap="binary", aspect="equal", interpolation="none")

    ax_clean.imshow(data, **imshow_args)
    ax_clean.set_title("Signal", fontsize="medium", color="darkgreen")

    ax_noise.imshow(data_noisy, **imshow_args)
    ax_noise.set_title("Input: Signal + Noise", fontsize="medium")

    for ax in (ax_clean, ax_noise):
        ax_remove_ticks(ax)
        ax.set_ylabel("x")
        ax.set_xlabel("t")

    for iwin, win_size in enumerate(window_sizes):
        if isinstance(win_size, int):
            win_size = (win_size, win_size)
        overlap = (int(win_size[0] / 2 - 1), int(win_size[1] / 2 - 1))

        ax_filt = fig.add_subplot(gs[1, iwin])
        ax_filt_norm = fig.add_subplot(gs[2, iwin])
        ax_spec = fig.add_subplot(gs[3, iwin])
        ax_spec.set_xscale("log")
        ax_spec.set_yscale("log")

        data_filtered = goldstein_filter(
            data_noisy,
            win_size,
            overlap=overlap,
            exponent=0.8,
            normalize_power=False,
        )
        data_filtered_norm = goldstein_filter(
            data_noisy,
            win_size,
            overlap=overlap,
            exponent=0.8,
            normalize_power=True,
        )
        ax_filt.imshow(data_filtered, **imshow_args)
        ax_filt_norm.imshow(data_filtered_norm, **imshow_args)

        for ax in (ax_filt, ax_filt_norm):
            ax.add_artist(
                patches.Rectangle((128, 128), *win_size, fc="w", lw=0, alpha=0.5)
            )

        signal_slice = slice(data.shape[1] // 2 + 100, data.shape[1])
        freqs = np.fft.rfftfreq(data[:, signal_slice].shape[1], DELTA_T)
        noise_idx = np.logical_and(noise_band[0] < freqs, freqs < noise_band[1])

        spec_noisy = get_norm_spec(data_noisy[:, signal_slice])
        spec_filtered = get_norm_spec(data_filtered[:, signal_slice])
        spec_filtered_norm = get_norm_spec(data_filtered_norm[:, signal_slice])
        spec_signal = get_norm_spec(data[:, signal_slice])

        noise_level_noisy = np.mean(spec_noisy[noise_idx])

        def get_db_improvement(spectrum):
            noise_level = spectrum[noise_idx].mean()
            return 10 * np.log10(noise_level / noise_level_noisy)

        ax_spec.plot(freqs, spec_noisy, c="k", alpha=0.3)
        ax_spec.plot(
            freqs,
            spec_filtered,
            c="firebrick",
            alpha=0.5,
            label=f"AFK ({get_db_improvement(spec_filtered):.1f} dB)",
        )
        ax_spec.plot(
            freqs,
            spec_filtered_norm,
            c="darkblue",
            label=f"NAFK ({get_db_improvement(spec_filtered_norm):.1f} dB)",
            alpha=0.5,
        )
        ax_spec.plot(freqs, spec_signal, c="darkgreen", alpha=0.3)

        ax_spec.axvspan(*noise_band, fc="#C6EBC9")
        ax_spec.set_title(f"{win_size[0]}x{win_size[1]} sample", fontsize="small")
        ax_spec.grid(alpha=0.3)
        ax_spec.set_ylim(1e-5, 2)
        ax_spec.set_xlabel("Frequency")
        ax_spec.spines["top"].set_visible(False)
        ax_spec.spines["right"].set_visible(False)
        ax_spec.legend(loc="lower left", fontsize="small")

        ax_remove_ticks(ax_filt)
        ax_remove_ticks(ax_filt_norm)

        if iwin == 0:
            ax_filt.set_ylabel("AFK filtered", color="firebrick")
            ax_filt_norm.set_ylabel("NAFK filtered", color="darkblue")
            ax_spec.set_ylabel("Norm. Log. Amplitude")

        for ax in (ax_filt, ax_filt_norm):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        ax_spec.set_yticklabels([])

    if filename:
        fig.savefig(filename, dpi=250)

    if show:
        plt.show()


size = (2048, 2048)
taper = np.bartlett(size[0])
field = get_sweeping_wavefield(
    size,
    freq_min=0.1,
    freq_max=2.0,
    amplitude=3.0,
    dx=1.0,
    fade_in=0.5,
)

if __name__ == "__main__":
    plot_filter_performance(
        field,
        noise_amp=0.1,
        filename="figures/synthetic_sweeping_wavefield.png",
        noise_band=(5.0, 50.0),
        show=True,
    )
