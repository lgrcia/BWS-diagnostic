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
#
#   Not fully documented


import lib.ops_processing as ops
import lib.diagnostic_tools as dt
from lib import measurement_simulations as ms
from other import lib_figures_M_thesis as lb

## PSB TUNNEL PROTOTYPE


folders = ["E:\BWS calibration (Processed)\TUNNEL BWS\PSB133rs__2017_03_10__19_27 PROCESSED",
           "E:\BWS calibration (Processed)\TUNNEL BWS\PSB133rs__2017_03_10__20_10 PROCESSED",
           "E:\BWS calibration (Processed)\TUNNEL BWS\PSB133rs__2017_03_08__15_56 PROCESSED",
           "E:\BWS calibration (Processed)\TUNNEL BWS\PSB133rs__2017_03_09__14_38 PROCESSED",
           "E:\BWS calibration (Processed)\TUNNEL BWS\PSB133rs__2017_03_09__15_46 PROCESSED"]

ps = ops.ProcessCalibrationResults(folders)
ps.run()
ps = ops.ProcessCalibrationResults(folders, reference_folder="E:\BWS calibration (Processed)\TUNNEL BWS")
ps.run()
dt.plot_multiple_calibrations_analysis(folders, in_or_out='IN', save=True, saving_name='C:\Rapport_images_eps/psb_tunnel_prototype_multiple_calibrations_analysis_in')
dt.plot_multiple_calibrations_analysis(folders, in_or_out='OUT', save=True, saving_name='C:\Rapport_images_eps/psb_tunnel_prototype_multiple_calibrations_analysis_out')
dt.plot_residuals_shape(folders, title='PSB tunnel prototype calibrations - 133 rad/s', save=True, saving_name='C:\Rapport_images_eps/psb_tunnel_prototype_residuals')


folder = "E:\BWS calibration (Processed)\TUNNEL BWS\PSB133rs__2017_03_10__19_27 PROCESSED"
dt.plot_calibration(folder, 'IN', separate=True, save=True, saving_name='C://Rapport_images_eps/psb_tunnel_prototype_in')
dt.plot_calibration(folder, 'OUT', separate=True, save=True, saving_name='C://Rapport_images_eps/psb_tunnel_prototype_out')
dt.plot_all_positions(folder, in_or_out='IN', save=False, saving_name='C://Rapport_images_eps/psb_tunnel_prototype_positions_in')
dt.plot_all_eccentricity(folder, in_or_out='IN', save=True, saving_name='C://Rapport_images_eps/psb_tunnel_prototype_eccentricities', separate=True)
dt.plot_all_speed(folder, in_or_out='IN', save=False, saving_name='C://Rapport_images_eps/psb_tunnel_prototype_speeds', separate=True)


# ## PSB LABORATORY PROTOTYPE
# ## VACUUM TEST
#
folders = ["E:\BWS calibration (Processed)\LAB\PSB55rs_Offset06_TableFixed__2017_04_21__11_54 PROCESSED",
           "E:\BWS calibration (Processed)\LAB\PSB55rs_Offset06_TableFixed_ShorterRail__2017_04_21__15_50 PROCESSED",
           "E:\BWS calibration (Processed)\LAB\PSB55rs_Offset06__2017_04_21__10_13 PROCESSED"]
ps = ops.ProcessCalibrationResults(folders)
ps.run()
ps = ops.ProcessCalibrationResults(folders, reference_folder="E:\BWS calibration (Processed)\LAB")
ps.run()
dt.plot_residuals_shape(folders, title='PSB laboratory prototype vacuum study calibrations - 55 rad/s', save=True, saving_name='C:\Rapport_images_eps/vacuum_study_55_residuals_shape')

folders = ["E:\BWS calibration (Processed)\LAB\PSB133rs_Offset06__2017_04_21__10_57 PROCESSED",
           "E:\BWS calibration (Processed)\LAB\PSB133rs_Offset06_TableFixed__2017_04_21__14_30 PROCESSED",
           "E:\BWS calibration (Processed)\LAB\PSB133rs_Offset06_TableFixed_ShorterRail__2017_04_21__16_38 PROCESSED"]
ps = ops.ProcessCalibrationResults(folders)
ps.run()
ps = ops.ProcessCalibrationResults(folders, reference_folder="E:\BWS calibration (Processed)\LAB")
ps.run()
dt.plot_residuals_shape(folders, title='PSB laboratory prototype vacuum study calibrations - 133 rad/s', save=True, saving_name='C:\Rapport_images_eps/vacuum_study_133_residuals_shape')

## FATIGUE TEST
#
folders = ["F:\pyBWS01\data - Copy\\133rs_CC__2017_06_01__10_45 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_01__17_50 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_02__17_37 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_06__10_57 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_06__17_08 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_07__09_50 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_07__17_51 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_08__09_57 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_08__17_14 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_09__09_51 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_09__17_13 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_12__10_30 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_13__08_57 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_13__17_32 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_14__09_30 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_14__17_10 PROCESSED",
           "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_15__09_26 PROCESSED"]


dt.plot_multiple_calibrations_analysis(folders, in_or_out='IN', save=True, saving_name='C:\Rapport_images_eps/psb_laboratory_prototype_multiple_calibrations_analysis_in')
dt.plot_multiple_calibrations_analysis(folders, in_or_out='OUT', save=True, saving_name='C:\Rapport_images_eps/psb_laboratory_prototype_multiple_calibrations_analysis_out')
dt.plot_residuals_shape(folders, title='fatigue test calibrations', save=True, saving_name='C:\Rapport_images_eps/fatigue_calibrations_residuals_shape')
dt.plot_global_residuals(folders, save=True, saving_name='C:\Rapport_images_eps/fatigue_test_overall_residuals')
dt.plot_calibration_procedure(folders=folders, origin=True, strategy=1, save=True, saving_name="C:\Rapport_images_eps/residuals_first_hypothesis")
dt.plot_calibration_procedure(folders=folders, origin=True, strategy=2, save=True, saving_name="C:\Rapport_images_eps/residuals_second_hypothesis")


folder = "E:\BWS calibration (Processed)\LAB\PSB133rs_Offset06__2017_04_21__10_57 PROCESSED"
dt.plot_calibration(folder, 'IN', separate=True, save=True, saving_name='C://Rapport_images_eps/psb_laboratory_prototype_in')
dt.plot_calibration(folder, 'OUT', separate=True, save=True, saving_name='C://Rapport_images_eps/psb_laboratory_prototype_out')


## SIMULATIONS

ms.Beam_width_error_over_different_calibration_curves("F:\pyBWS01\data - Copy\\mean_fit.mat", folders, save=True, saving_name="C:\Rapport_images_eps/relative_error_for_different_calibration_curves_15um", position_random_error=15e-6)
ms.Beam_width_error_over_different_calibration_curves("F:\pyBWS01\data - Copy\\mean_fit.mat", folders, save=True, saving_name="C:\Rapport_images_eps/relative_error_for_different_calibration_curves_5um", position_random_error=5e-6)
ms.simulation_of_beam_width_measurements_error("F:\pyBWS01\data - Copy\\mean_fit.mat", "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_02__17_37 PROCESSED", save=False, saving_name="C:\Rapport_images_eps/relative_error_vs_pps_15um", position_random_error=15e-6, theoretical_curve=True)
ms.simulation_of_beam_width_measurements_error("F:\pyBWS01\data - Copy\\mean_fit.mat", "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_02__17_37 PROCESSED", save=False, saving_name="C:\Rapport_images_eps/relative_error_vs_pps_5um", position_random_error=5e-6, theoretical_curve=True)
ms.simulate_wire_profile_measurements("F:\pyBWS01\data - Copy\\mean_fit.mat", "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_02__17_37 PROCESSED",
                                      gauss_beam_position=5.45e-3, gauss_beam_sigma=150e-6, diagnostic=True,
                                      save=True, saving_name="C:\Rapport_images_eps/1000_gaussian_beam_pps_05",
                                      position_random_error=15e-6)
ms.simulate_wire_profile_measurements("F:\pyBWS01\data - Copy\\mean_fit.mat", "F:\pyBWS01\data - Copy\\133rs_CC__2017_06_02__17_37 PROCESSED",
                                      gauss_beam_position=5.45e-3, gauss_beam_sigma=1000e-6, diagnostic=True,
                                      save=True, saving_name="C:\Rapport_images_eps/1000_gaussian_beam_pps_251",
                                      position_random_error=15e-6)
lb.signal_types()
lb.camel_backs()
lb.algorithm_study()
lb.single_reflection_study()
lb.laser_study()
lb.laser_simulations()
lb.signal_error(case=0)
lb.signal_error(case=1)
lb.signal_error(case=2)
lb.scan_outome()
lb.beam_displacer()
lb.plot_calibration_procedure()
lb.theoretical_movement(save=True, saving_name="C:/Rapport_images_eps/ideal_wire_movement")

folder = "E:\BWS calibration (Processed)\LAB\\Not_Vacuum_300Scans_133rs__2017_05_22__13_40 PROCESSED"
dt.plot_calibration(folder, 'IN', separate=True, save=True, saving_name='C://Rapport_images_eps/psb_laboratory_prototype_in_static')
dt.plot_calibration(folder, 'OUT', separate=True, save=True, saving_name='C://Rapport_images_eps/psb_laboratory_prototype_out_static')
