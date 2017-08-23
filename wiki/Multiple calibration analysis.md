# Single calibration using BWS-diagnostic library

A single calibration anlalysis is done with the functions:

* [`plot_multiple_calibrations_analysis`](**`plot_calibration`**)
* [`plot_global_residuals`](**`plot_all_positions`**)
* [`plot_residuals_shape`](`plot_all_speed`)




#### **`plot_multiple_calibrations_analysis`** :

Plot all the residuals (with respect to the calibration curve when doing the processing in [Multiple calibration analysis](Multiple cqlibrqtion analysis) ) and their histogram

* **attributes**:
  * `folder_name` _string_ : name of the **PROCESSED folder**
  * `in_or_out` _string_ : _'IN'_ or _'OUT'_ to study IN-scans or OUT-scans
  * `save` _(optional) boolean_ : save or not the figure(s)
  * `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved

Example:

```python
from lib import diagnostic_tools as dt

PROCESSED_folders = ["D:PROCESSED_folder_1",
                     "D:PROCESSED_folder_2",
                     "D:PROCESSED_folder_3",
                     "D:PROCESSED_folder_4",
                     ...]

# Plot the analysis of multiple calibrations
dt.plot_multiple_calibrations_analysis(folders=PROCESSED_folders, in_or_out='IN')
```
![psb_laboratory_prototype_multiple_calibrations_analysis_in](/Users/lgr/Documents/BWS-diagnostic wiki/psb_laboratory_prototype_multiple_calibrations_analysis_in.png)



#### **`plot_global_residuals`** :

Plot all the residuals (with respect to the calibration curve when doing the processing in [Multiple calibration analysis](Multiple cqlibrqtion analysis) ) as if considered comming from the same calibration. Plot an histogram of the complete data

- **attributes**:
  - `folder_name` _string_ : name of the **PROCESSED folder**
  - `save` _(optional) boolean_ : save or not the figure(s)
  - `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved

Example:

```python
from lib import diagnostic_tools as dt

PROCESSED_folders = ["D:PROCESSED_folder_1",
                     "D:PROCESSED_folder_2",
                     "D:PROCESSED_folder_3",
                     "D:PROCESSED_folder_4",
                     ...]

# Plot all the residuals in one plot as if they were part of the same calibration
dt.plot_global_residuals(folders=PROCESSED_folders)

```

![fatigue_test_overall_residuals](/Users/lgr/Documents/BWS-diagnostic wiki/fatigue_test_overall_residuals.png)

#### **`plot_residuals_shape`** :

Plot the residuals shape of different calibration (residuals are computed with respect to the calibration curve when doing the processing in [Multiple calibration analysis](Multiple cqlibrqtion analysis) )

**attributes**:

- - `folder_name` _string_ : name of the **PROCESSED folder**
  - `save` _(optional) boolean_ : save or not the figure(s)
  - `saving_name`  _(optional) string_ : path where the figure(s) are going to be saved

Example:

```python
from lib import diagnostic_tools as dt

PROCESSED_folders = ["D:PROCESSED_folder_1",
                     "D:PROCESSED_folder_2",
                     "D:PROCESSED_folder_3",
                     "D:PROCESSED_folder_4",
                     ...]

# Plot all the residuals in one plot as if they were part of the same calibration
dt.plot_global_residuals(folders=PROCESSED_folders)
```

![fatigue_calibrations_residuals_shape](/Users/lgr/Documents/BWS-diagnostic wiki/fatigue_calibrations_residuals_shape.png)