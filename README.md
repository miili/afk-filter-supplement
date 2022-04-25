# Electronic Supplement Plotting Scripts

> De-noising distributed acoustic sensing data using an adaptive frequency-wavenumber filter; Marius P Isken, S Heimann, H Vasyura-Bathke, T Dahm;

Electronic supplement for doi.org/

## Abstract

Data recorded by distributed acoustic sensing (DAS) along an optical fiber sample the spatial and temporal properties of seismic wavefields at high spatial density. This lead to massive data when collected for seismic monitoring along kilometer long cables. The spatially coherent signals from weak seismic arrivals within the data are often obscured by incoherent noise. We present a flexible and computationally efficient filtering technique which makes use of the dense spatial and temporal sampling of the data and can handle the large amount of data. The presented adaptive frequency-wavenumber filter suppresses the incoherent seismic noise while amplifying the coherent wave field. We analyse the response of the filter in time and spectral domain, and we demonstrate its performance on a noisy data set that was recorded in a vertical borehole observatory showing active and passive seismic phase arrivals. In these data we can suppress the noise up to 20 dB. Lastly, we present a performant open-source software implementation enabling real-time filtering of large DAS data sets.

![DAS filtered](https://github.com/miili/afk-filter-supplement/raw/master/figures/gfz2020wswf-AFK.png)

## Distributed Acoustic Sensing Data

The different DAS data sets are located in `data/`

1. VSP shot at 200 m distance: `das-data-vsp.npy`, shown in Figure 1 and Figure 2.

2. Regional earthquake M=4.0: `data-DAS-gfz2020wswf.npy`, shown in Figure S1 and Figure S2.

3. Local earthquake Ml=1: `landwuest_UTC_20210422_034648.386.tdms`, shown in Figure S4.

## Plotting scripts

The plotting scripts require pyrocko and the package lightguide.
