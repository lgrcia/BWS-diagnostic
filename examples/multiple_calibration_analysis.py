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

PROCESSED_folders = ["F:\pyBWS01\data\\133rs_CC__2017_06_08__17_14 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_09__09_51 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_09__17_13 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_12__10_30 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_13__08_57 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_13__17_32 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_14__09_30 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_14__17_10 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_15__09_26 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_01__10_45 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_01__17_50 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_02__17_37 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_06__10_57 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_06__17_08 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_07__09_50 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_07__17_51 PROCESSED",
                     "F:\pyBWS01\data\\133rs_CC__2017_06_08__09_57 PROCESSED"]

# #
# Plot the analysis of multiple calibrations
dt.plot_multiple_calibrations_analysis(folders=PROCESSED_folders, in_or_out='IN')
#
# Plot all the residuals in one plot as if they were part of the same calibration
dt.plot_global_residuals(folders=PROCESSED_folders)

# Plot the residuals of all the calibration in the same plot
dt.plot_residuals_shape(folders=PROCESSED_folders)
#
