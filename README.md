# ASL-Sequential-Dataset
(Mixed motion of daily activities and ASL signs)

This dataset contains sequential motions of 3 daily activities (walking, sitting, standing up) and 15 ASL signs. The subjects performed the prompted activities for 24.2 seconds in the line of sight of the radar. Complete description of sequences is listed below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/66335348/123162029-bc279d00-d435-11eb-914b-06c5b20a4489.png" />
</p>

The raw data is acquired using TI’s AWR1642BOOST radar and DCA1000EVM data capture card.

### Directory Structure:

The main folder (ASL Sequential Dataset) contains 2 subfolders for different ASL user groups (non-ASL users and ASL users). While non-ASL users are hearing participants, who were trained only before the experiment, ASL-users are either child-of-deaf-adults (CODAs) or ASL learners. Under each participant group there are multiple subfolders corresponding to different subjects (e.g., 12 jan ozgur mahbub ladi emre folder has data of 4 people). Under each subfolder there are 5 folders (envelopes, labels, microDoppler, rangeDOA and rangeDoppler). Here, ‘microDoppler’, ‘rangeDOA’ and ‘rangeDoppler’ folders contain different input representations obtained from complex raw data files. ‘labels’ folder contains label files for each .png or .avi files under ‘microDoppler’, ‘rangeDOA’ and ‘rangeDoppler’ folders. .txt files contains the labels for each time step (column for images, frame for videos) of the input representations. Since rangeDOA maps and rangeDoppler maps has the same number of frames, .txt files under ‘labels/rangeDoppler’ folder can be used for both rangeDoppler and rangeDOA ground truth. Classes corresponding to integer label numbers are as follows:

0: No motion, 1: walking, 2: sitting, 3: standing up, 4: TIRED, 5: BOOK, 6: SLEEP, 7: EVENING, 8: READY, 9: HOT, 10: HOT, 11: MONTH, 12: COOK, 13: AGAIN, 14: SUMMON, 15: MAYBE, 16: NIGHT, 17: SOMETHING, 18: TEACHER, 19: TEACH.

Length of .txt files will be the same as number of columns of images or number of frames of videos, and they will share the same file name except the extension (i.e., replace .png and .avi with .txt).

‘envelopes’ folder contains the Euclidean distance between upper and lower envelopes of the microDoppler spectrograms which can be used to detect where the motions start and end (i.e., motion detection).

4th character of each file name indicates the SEQUENCE number given in the table above. For example, the file ‘11040000_1618505108_1.png’ belongs to Seq. 4 in the table.

Should you have any questions or need to access the raw data files to apply your own algorithms please contact Dr. Sevgi Zubeyde Gurbuz (szgurbuz@ua.edu).

###### [ci4r.ua.edu](https://ci4r.ua.edu)
