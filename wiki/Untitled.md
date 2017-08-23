# BWS diagnostic tool documentation

**pyProtoBWS** is a tool to analyse the raw data coming from the new generation of Beam Wire Scanners at CERN (BE-BI-PM). 

The goal of this tool is to **assist, diagnose and asses the performances** of the different prototypes. It also aims to study the calibration needs of the new scanners.

Before any complete documentation, here is a good way to start the data analysis : 
- [*First step* : **process the TDMS** files coming from the calibration bench](#first_step)
- [*Second step* : Processing validation and **first calibration data**](#second_step)
- [*Final step* : Processing of several calibrations to asses the **scanner performances**](#final_step)

## <a name="first_step"></a> *First step* : process the TDMS files coming from the calibration bench


The processing of the TDMS files will be done through a parameter file. This .cgf file (actually a .txt file) can be modified to adapt the processing parameters later. A complete description of this file is given [here](parameters.cfg)

Once this file is ready the next step is to write some lines of code : 
```python
import lib
from lib import ops_processing as ops

TDMS_folder  = 'C:/your_path/133rs_test_folder'

# Running this line is recommended to properly evaluate the time and space required by the processing
ops.print_raw_data_analysis(TDMS_folder)

#Complete processing; go take a cup of coffee...
ops.complete_processing(TDMS_folder, 'C:/another_path/destination_folder')
```

After some processing time the results can be found in *C:/another_path/destination_folder*. Do not delete the *RAW_DATA* folder temporarily created during the processing phase.

*C:/another_path/destination_folder* will contain the following folder :

**`133rs_test_folder PROCESSED`** containing: 

- `PROCESSED_IN.mat` :

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

- `PROCESSED_OUT.mat` : The same as `PROCESSED_IN.mat` but for OUT scans

## <a name="second_step"></a> *Second step* : Processing validation and **first calibration data**
After the processing is done a validation phase in necessary to ensure that the data coming from the OPS have been well processed (no slits jumps)


Here is a brief description of all the plotting tools available to treat the processed data of a complete calibration :

`plot_calibration(folder_name, in_or_out)` : <br/> plot of a complete calibration for IN or OUT scans

`plot_calibration_in_and_out(folder_name)` : <br/> plot of a complete calibration both for IN and OUT scans

`plot_all_speed(folder_name, in_or_out)` : <br/> plot of all speeds with behavioral statistics

`plot_all_eccentricity(folder_name, in_or_out)` : <br/>plot of all eccentricity with behavioral statistics **[Useful for slits jump analysis]**

`plot_references_divergence(folder_name, in_or_out)` : <br/> plot of the reference detection in time over the complete calibration **[Useful for disk slipping analysis]**|

For example : 
```python
import lib
from lib import diagnostic_tools as dt

processed_folder  = 'C:/another_path/133rs_test_folder PROCESSED'

dt.plot_calibration(processed_folder, 'IN')
```

Will produce the following plot : 

![](https://github.com/LionelGarcia/BWS_diagnostic/blob/master/doc_images/figure_1.png)

**Plot produced with *Matplotlib* and [Prairie](https://github.com/LionelGarcia/Prairie) style**

## <a name="final_step"></a> *Final step* : Processing of several calibrations to asses the **scanner performances**

The final step correspond to performance assessment of the scanner and define the performances of the instrument regarding its calibration.

One of the key component of this part is the study of several calibrations with some of the following functions : 
- `plot_calibration(folder_name, in_or_out)` : <br/> plot of a complete calibration for IN or OUT scans

For example : 
```python
import lib
from lib import diagnostic_tools as dt

folders = {'../data/133_CC__2017_06_12__17_29 PROCESSED',
           '../data/133rs_CC__2017_06_01__10_45 PROCESSED',
           '../data/133rs_CC__2017_06_01__17_50 PROCESSED',
           '../data/133rs_CC__2017_06_02__17_37 PROCESSED',
           '../data/133rs_CC__2017_06_06__10_57 PROCESSED',
           '../data/133rs_CC__2017_06_06__17_08 PROCESSED',
           '../data/133rs_CC__2017_06_07__09_50 PROCESSED',
           '../data/133rs_CC__2017_06_07__17_51 PROCESSED',
           '../data/133rs_CC__2017_06_08__09_57 PROCESSED',
           '../data/133rs_CC__2017_06_08__17_14 PROCESSED',
           '../data/133rs_CC__2017_06_09__09_51 PROCESSED',
           '../data/133rs_CC__2017_06_09__17_13 PROCESSED',
           '../data/133rs_CC__2017_06_12__10_30 PROCESSED',
           '../data/133rs_CC__2017_06_13__08_57 PROCESSED',
           '../data/133rs_CC__2017_06_13__17_32 PROCESSED',
           '../data/133rs_CC__2017_06_14__09_30 PROCESSED',
           '../data/133rs_CC__2017_06_15__09_26 PROCESSED'}

redsiduals_comparision_from_origin(folders)
```

Will produce the following results : 

![](https://github.com/LionelGarcia/BWS_diagnostic/blob/master/doc_images/figure_3.png)

**Plot produced with *Matplotlib* and [Prairie](https://github.com/LionelGarcia/Prairie) style**


BWS_diagnostic is a tool to analyse the raw data coming from the new generation of Rotational Beam Wire Scanners at CERN (BE-BI-PM).

Home
Installation:

Installation and use
Processing:

Calibration processing
Single calibration analysis
Multiple calibration analysis
Scan raw data analysis
Parameters:

Algorithm parameters
