# Single calibration using BWS-diagnostic library

A single calibration anlalysis is done with the functions:

* [`plot_calibration`](**`plot_calibration`**)
* [`plot_all_positions`](**`plot_all_positions`**)
* [`plot_all_speed`](`plot_all_speed`)
* [`plot_all_eccentricity`](`plot_all_eccentricity`)



#### **`plot_calibration`** :

* **attributes**:
  * `folder_name` _string_ : name of the **PROCESSED folder**
  * `in_or_out` _string_ : _'IN'_ or _'OUT'_ to study IN-scans or OUT-scans
  * `save` _(optional) boolean_ : save or not the figure(s)
  * `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved
  * `separate`  _(optional) boolean_ : save calibration curve and residuals plot in two different figures
  * `complete_residuals_curve` _(optional) boolean_ : see a plot with residuals vs. scans (the normal plot is vs. laser position)

Example:

```python
from lib import diagnostic_tools as dt

PROCESSED_folder = "C:/Your_path"

# Complete calibration plot
dt.plot_calibration(folder_name=PROCESSED_folder, in_or_out='IN', complete_residuals_curve=True)
```



#### **`plot_all_positions`** :

Plot all the postions profile of the calibration

- **attributes**:
  - `folder_name` _string_ : name of the **PROCESSED folder**
  - `in_or_out` _string_ : _'IN'_ or _'OUT'_ to study IN-scans or OUT-scans
  - `save` _(optional) boolean_ : save or not the figure(s)
  - `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved
  - `separate`  _(optional) boolean_ : save calibration curve and residuals plot in two different figures

Example:

```python
from lib import diagnostic_tools as dt

PROCESSED_folder = "C:/Your_path"

# All position profile plot (may be long)
dt.plot_all_positions(folder_name=PROCESSED_folder, in_or_out='OUT')
```



#### **`plot_all_speeds`** :

Plot all the speed profile of the calibration, the residuals of these profile with regard to their mean and some statistics

- **attributes**:
  - `folder_name` _string_ : name of the **PROCESSED folder**
  - `in_or_out` _string_ : _'IN'_ or _'OUT'_ to study IN-scans or OUT-scans
  - `save` _(optional) boolean_ : save or not the figure(s)
  - `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved
  - `separate`  _(optional) boolean_ : save calibration curve and residuals plot in two different figures

Example:

```python
from lib import diagnostic_tools as dt

PROCESSED_folder = "C:/Your_path"

# All speed profiles plot (may be long)
dt.plot_all_speed(folder_name=PROCESSED_folder, in_or_out='OUT')
```



#### **`plot_all_eccentricities`** :

Plot all the eccentricity profile of the calibration, the residuals of these profiles with regard to their mean and some statistics

- **attributes**:
  - `folder_name` _string_ : name of the **PROCESSED folder**
  - `in_or_out` _string_ : _'IN'_ or _'OUT'_ to study IN-scans or OUT-scans
  - `save` _(optional) boolean_ : save or not the figure(s)
  - `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved
  - `separate`  _(optional) boolean_ : save calibration curve and residuals plot in two different figures

Example:

```Python
from lib import diagnostic_tools as dt

PROCESSED_folder = "C:/Your_path"

# All eccentricity profiles plot (may be long)
dt.plot_all_eccentricity(folder_name=PROCESSED_folder, in_or_out='OUT')
```



#### **`plot_RDS`** :

RDS plot of the complete calibration and for each scan

- **attributes**:
  - `folder_name` _string_ : name of the **PROCESSED folder**
  - `in_or_out` _string_ : _'IN'_ or _'OUT'_ to study IN-scans or OUT-scans
  - `save` _(optional) boolean_ : save or not the figure(s)
  - `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved
  - `offset`  _(optional) int_ : number of RDS points to cut at the begining and the end of the overall plot

Example:

```Python
from lib import diagnostic_tools as dt

PROCESSED_folder = "C:/Your_path"

# Relative distance signature plot (may be long)
dt.plot_RDS(folder_name=PROCESSED_folder, in_or_out='OUT')
```



# Single calibration using BWS-diagnostic GUI

1. Go to the upper tab `Single calibration analysis`
2. Choose a folder in `Processed folder`
   * Information will be loaded and displayed to the left 
3. Pess `Import`



After a loading time when the programm is frozen, multiple tabs are displaying the **calibration information**:

* **Calibration plot** IN and OUT scans (fit of the wire movement curve + residuals and histogram)
* **RDS plot** for IN and OUT scans

Then the table to the right give acces to each indidual scan an update **a the selected scan** the:

* **Speed profile** for OPS A and B, IN and OUT
* **Eccentricity profile** for OPS A and B, IN and OUT



Finnaly, by completing the `TDMS folder` section with the original TDMS folder of the PROCESSED calibration, the progrmam allows to reprocess a single scan the same way as in [Scan raw data anlysis](Scan raw data anlysis)