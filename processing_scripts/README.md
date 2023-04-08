# Processing scripts to generate input representations (micro-Doppler spectrograms, RD and RA maps and spectrogram envelopes)

`data_collect_trigger_v2.m` is the top script to generate all input representations except the envelopes (use `Find_envelopes.m` to extract spectrogram envelopes).

All the input/output paths in the main scripts should be replaced with your own custom directory. Each recording consists of 5 repetition of a sequence and is split
into 2 1gb and 1 180mb raw data files (\*.bin). Hence, `data_collect_trigger_v2.m` first extracts the radar data cubes (RDCs) of each raw data file belonging to the 
same recording, then concatenates the extracted RDCs. After that, it divides the whole RDC into 5 equal segments corresponding to each repetition, and calls other 
helper functions to generate different input representations. 
