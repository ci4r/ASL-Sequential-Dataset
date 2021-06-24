## Signal processing scripts to process raw RF data.

- **data_collect_trigger_v2.m** is the main script to read raw .bin files and call other functions.
- **RDC_extract.m** reads .bin files and returns the 3D radar data cubes (RDCs) with shapes of (num. of ADC samples, num. of chirps (pulses), num. of channels).
- **RDC_to_rangeDopp.m** applies 2D Fast Fourier Transform (FFT) on RDCs and creates range-Doppler (RD) maps, and **ca_cfar.m** applies CA-CFAR to find range limits, and returns them.
- **RDC_to_microDopp.m** and **myspecgramnew.m** applies short-time Fourier Transform (STFT) to generate micro-Doppler spectrograms.
- **RDC_to_rangeDOA_AWR1642.m** applies MUSIC algorithm with Optical Flow enhancement, and creates range-Angle (RA) maps.
- **Find_envelopes.m** and **env_find.m** extracts upper and lower envelopes of spectrograms and computes the Euclidean distance between them.
- **label_crop_fig2.m** can be used to annotate spectrograms and RA and RD maps in temporal domain.
- **similarity.m** performs the fidelity analysis between ASL and non-ASL users. 


