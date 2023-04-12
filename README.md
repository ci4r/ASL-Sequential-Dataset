# ASL-Sequential-Dataset (77 GHz FMCW MIMO)
(Mixed motion of daily activities and ASL signs)

## Cite as

*E. Kurtoğlu, A. C. Gurbuz, E. A. Malaia, D. Griffin, C. Crawford and S. Z. Gurbuz, "ASL Trigger Recognition in Mixed Activity/Signing Sequences for RF Sensor-Based User Interfaces," in IEEE Transactions on Human-Machine Systems, vol. 52, no. 4, pp. 699-712, Aug. 2022, doi: 10.1109/THMS.2021.3131675..*

This dataset contains sequential motions of 3 daily activities (walking, sitting, standing up) and 15 ASL signs. The subjects performed the prompted activities for 24.2 seconds in the line of sight of the radar. Complete description of sequences is listed below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/66335348/123162029-bc279d00-d435-11eb-914b-06c5b20a4489.png" />
</p>

The raw data is acquired using TI’s AWR1642BOOST radar and DCA1000EVM data capture card. A total of 200 hearing participant (Non-ASL user) samples and 94 native (ASL user) participant samples for each sequence are acquired.

### Directory Structure:

The main folder (ASL Sequential Dataset) contains 2 subfolders for different ASL user groups (non-ASL users and ASL users). While non-ASL users are hearing participants, who were trained only before the experiment, ASL-users are either child-of-deaf-adults (CODAs) or ASL learners. Under each participant group there are multiple subfolders corresponding to different subjects (e.g., 12 jan ozgur mahbub ladi emre folder has data of 4 people). Under each subfolder there are 5 folders (envelopes, labels, microDoppler, rangeDOA and rangeDoppler). Here, `microDoppler`, `rangeDOA` and `rangeDoppler` folders contain different input representations obtained from complex raw data files. `labels` folder contains label files for each .png or .avi files under `microDoppler`, `rangeDOA` and `rangeDoppler` folders. .txt files contain the labels for each time step (column for images, frame for videos) of the input representations. Since rangeDOA maps and rangeDoppler maps has the same number of frames, .txt files under `labels/rangeDoppler` folder can be used for both rangeDoppler and rangeDOA ground truth. Length of the .txt files will be the same as number of columns of images or number of frames of videos, and they will share the same file name except the extension (i.e., replace .png and .avi with .txt). Classes corresponding to integer label numbers are as follows:

0: No motion, 1: walking, 2: sitting, 3: standing up, 4: TIRED, 5: BOOK, 6: SLEEP, 7: EVENING, 8: READY, 9: HOT, 10: HOT, 11: MONTH, 12: COOK, 13: AGAIN, 14: SUMMON, 15: MAYBE, 16: NIGHT, 17: SOMETHING, 18: TEACHER, 19: TEACH.

A sample microDoppler spectrogram belonging to SEQUENCE 5 is shown below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/66335348/142496455-000a6f49-e945-43f2-848c-c382a125a768.png" />
</p>

`envelopes` folder contains the Euclidean distance between upper and lower envelopes of the microDoppler spectrograms as a time-series vector which can be used to detect where the motions start and end (i.e., motion detection). A sample envelope extraction process is as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/60670979/142490804-448d5d83-a6bc-4619-9ae5-4a2cb376392a.png" />
</p>

4th index of each file name indicates the SEQUENCE number given in the table above. For example, the file ‘11040000_1618505108_1.png’ belongs to Seq. 4 in the table. The last index after `_` indicates the iteration number \[1-5\].

### Raw Data and Pre-processing

Raw data of the RF measurements are also available under `Raw data/ASL Sequential Raw Files.zip` (~1.3 TB). Top scripts `data_collect_trigger_v2.m` and `data_collect_trigger_v3_revision.m` can be used to create the different input representations. Each recording consists of 5 repetitions of a sequence and is split into 3 .bin files (1gb, 1gb, 180mb). The aforementioned top scripts first reads these 3 data files and extract the radar data cube (RDC) of each of them. Then concatenate them in slow-time dimension, and split them into 5 sequences. Splitted sequences are then processed individually and different input representations are generated. Note that extraction of envelopes are done explicitly by `Find_envelopes.m` script.

Should you have any questions about the dataset, please contact Dr. Sevgi Zubeyde Gurbuz (szgurbuz@ua.edu).

#### [ci4r.ua.edu](https://ci4r.ua.edu)

<!---
## [DATASET DOWNLOAD](https://bama365-my.sharepoint.com/:f:/g/personal/ekurtoglu_crimson_ua_edu/EjrqvtrhIupDrPnefNwOLyIBJn80gv_6UlFAgWjku2srOw?e=Fkx05A)
--->

<!--- (https://drive.google.com/drive/folders/1AZRB-uCphFzmG-q_0cvmIad1un4HPMxU?usp=sharing) --->

Alternative download link:
## [DATASET DOWNLOAD](https://storage.cloud.google.com/asl_sequential_thms2021_dataset)
