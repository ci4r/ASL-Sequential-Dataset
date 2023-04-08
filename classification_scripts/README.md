# ASL Trigger Recognition in Mixed Activity/Signing Sequences for RF Sensor-Based User Interfaces

While the MATLAB files under `processing_scripts` are used for raw data pre-processing to generate `range-Doppler`, `range-Azimuth` maps and `micro-Doppler` spectrograms, Python scripts are used for classification.

`classification_scripts/create_dataset*.ipynb` files are used to read the created videos and images and save them as pickle or .hdf5 files.

`classification_scripts/Final baseline 2.ipynb` is the main classification script which implements the STA/LTA motion detector and the JD-MTML model explained in the original paper.
