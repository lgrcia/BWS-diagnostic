##  Processing parameters

The first step before processing any file is to setup the parameters with values making sense for the processing. Please read [Algorithm and parameters](Algorithm and parameters) before doing any processing.

##  Processing using BWS-diagnostic library

During a complete calibration the BWS realise multiple scan and the data comming form the calibration bench (OPS A, OPS B and the photodiode) is saved as **[tdms](http://www.ni.com/white-paper/3727/en/)** files into a **tdms folder**. 

The processing start by localising the tdms folder and entering it as an input of the calibration processing function:

```python
from lib import ops_processing as ops

TDMS_folders = ["D:/your_tdms_folder"]

destination_folder = "D:/your_destination_path"

# Single/Multiple calibration processing
for TDMS_folder in TDMS_folders:
    ops.process_complete_calibration(TDMS_folder, destination_folder)
```

The folowing code will :

1. Convert the **.tdms** files in **.mat** files
2. Process the IN-scans
3. Process the OUT-scans
4. Save the results in `D:/your_destination_path/your_tdms_folder_PROCESSED `containig:
   * `PROCESSED_IN.mat` :

| Name                    | Size  | Class    | Description                              |
| ----------------------- | ----- | -------- | ---------------------------------------- |
| **angular_position_SA** | **n** | *cell*   | Processed angular positions of sensor A for each scans |
| **angular_position_SB** | **n** | *cell*   | Processed angular positions of sensor B for each scans |
| **data_PD**             | 0     | *double* | Raw photodiode data for each scans       |
| **eccentricity**        | **n** | *cell*   | Computed eccentricty for each scans      |
| **laser_position**      | **n** | *double* | LabView values of the laser position (to convert into real laser position) |
| **occlusion_position**  | **n** | *double* | Angular position of the fork at laser crossing |
| **scan_number**         | **n** | *cell*   | Scan index (useful for several scan at the same laser position) |
| **speed_SA**            | **n** | *double* | Computed angular speed using sensor A data |
| **speed_SB**            | **n** | *int32*  | Computed angular speed using sensor B data |
| **time_SA**             | **n** | *cell*   | Time reference for Sensor A angular positions |
| **time_SB**             | **n** | *cell*   | Time reference for Sensor B angular positions |

with ***n*** the total number of scans

* `PROCESSED_OUT.mat` : The same as `PROCESSED_IN.mat` but for OUT scans
* `Calibration_results.mat`: Containing a lot of calibration info

These files will then be used to [analyse the calibration data](Complete calibration processing)

 ## Processing using BWS-diagnostic GUI

Another way to process the data is to use the **BWS-diagnostic expert application**. It has the advantage of not using any code and allows to **test the processing with different parameters** before launching a complete calibration processing. On the other hand you can't process several calibration at once (which in something to avoid in order to adapt each parameters sets for the different calibrations)

Follow the steps:

1. Go on the upper tab `Caibration processing`
2. Choose your tdms folder in `TDMS folder`
3. Choose your destination folder in `Destination folder`
4. Test the processing with the actual `parameter.cfg` file in `Test processing`
5. Adjust the parameters by opening `parameter.cfg` by clicking `Set parameters`
   * Repeat step 4. et 5. until the desired processing in reached
6. Click on `Process`
7. See directly the results by clicking `Show PROCESSED data` or follow the [single calibration analysis](Single calibration analysis)

And you are done :+1:

