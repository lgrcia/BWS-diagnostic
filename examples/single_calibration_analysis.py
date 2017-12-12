#   --------------------------------------------------------------------------
# Copyright (c) <2017> <Lionel Garcia>
# BE-BI-PM, CERN (European Organization for Nuclear Research)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#   --------------------------------------------------------------------------


from lib import diagnostic_tools as dt



# BWS PROTOTYPE SN64
# ------------------
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_tunnel_prototype (Python processed mat - PROCESSED folder)\PSB133rs__2017_03_10__20_10 PROCESSED"
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_tunnel_prototype (Python processed mat - PROCESSED folder)\PSB133rs__2017_03_10__19_27 PROCESSED"
PROCESSED_folder = "G:\Projects\BWS_Calibrations\Calibrations\BWS_sn64\ProcessedData\SN64__2017_12_11__17_29_Short PROCESSED"


# BWS PROTOTYPE SN66
# ------------------
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_laboratory_ptotoype (Python processed mat - PROCESSED folder)\PSB133rs_Offset06__2017_04_21__10_57 PROCESSED"
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_laboratory_ptotoype (Python processed mat - PROCESSED folder)\PSB133rs_Offset06_TableFixed__2017_04_21__14_30 PROCESSED"

# This one!
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_laboratory_ptotoype (Python processed mat - PROCESSED folder)\PSB133rs_Offset06_TableFixed_ShorterRail__2017_04_21__16_38 PROCESSED"


# BWS PROTOTYPE SN65
# ------------------
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_tunnel_prototype_nbr2 (Python processed mat - PROCESSED folder)\CC01__2017_09_14__16_35 PROCESSED"

# This one!
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_tunnel_prototype_nbr2 (Python processed mat - PROCESSED folder)\CC05__2017_09_15__15_57 PROCESSED"

#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_tunnel_prototype_nbr2 (Python processed mat - PROCESSED folder)\PC01__2017_09_14__17_11 PROCESSED"
#PROCESSED_folder = "E:\BWS calibration - data\BWS calibration (Processed)\PSB_tunnel_prototype_nbr2 (Python processed mat - PROCESSED folder)\PC03__2017_09_15__16_41 PROCESSED"

# BWS PROTOTYPE PS SN128
# ----------------------
#PROCESSED_folder = "G:\Projects\BWS_Calibrations\Calibrations\PS_BWS_SN128\ProcesedData\S128__2017_11_27__16_26 PROCESSED"
ParametersCurve = []

# Complete calibration plot
#dt.plot_calibration(folder_name=PROCESSED_folder, in_or_out='IN', complete_residuals_curve=True)
#dt.plot_calibration_INOUT(folder_name=PROCESSED_folder,complete_residuals_curve=False, remove_sytematics=False, N = 20, impose_parameters=False, parameters = ParametersCurve, inout_independent=True)

# All position profile plot (may be long)
#dt.plot_all_positions(folder_name=PROCESSED_folder, in_or_out='IN')

# All eccentricity profiles plot (may be long)
#dt.plot_all_eccentricity(folder_name=PROCESSED_folder, in_or_out='OUT')

# All speed profiles plot (may be long)
#dt.plot_all_speed(folder_name=PROCESSED_folder, in_or_out='OUT')

# Relative distance signature plot (may be long)
#dt.plot_RDS(folder_name=PROCESSED_folder, in_or_out='OUT')

# References detection in time VS scan number
#dt.plot_all_referencedetections(folder_name=PROCESSED_folder, in_or_out='OUT', timems = 400)
