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

PROCESSED_folder = "C:/Your_path"

# Complete calibration plot
dt.plot_calibration(folder_name=PROCESSED_folder, in_or_out='IN', complete_residuals_curve=True)

# All position profile plot (may be long)
dt.plot_all_positions(folder_name=PROCESSED_folder, in_or_out='OUT')

# All eccentricity profiles plot (may be long)
dt.plot_all_eccentricity(folder_name=PROCESSED_folder, in_or_out='OUT')

# All speed profiles plot (may be long)
dt.plot_all_speed(folder_name=PROCESSED_folder, in_or_out='OUT')

# Relative distance signature plot (may be long)
dt.plot_RDS(folder_name=PROCESSED_folder, in_or_out='OUT')
