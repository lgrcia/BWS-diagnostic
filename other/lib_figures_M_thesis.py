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


import random
import numpy as np
import configparser
import scipy.io as sio

from matplotlib import mlab
from nptdms import TdmsFile
from scipy.stats import norm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from lib import utils
from lib import prairie
from lib import ops_processing as ops

DPI = 200


# Other

def signal_types():
    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    file = "E:\BWS_Fatigue_Tests\Calibration_Bench_Files\\133rs_CC__2017_06_14__09_30\CALI__P37.00__S1____2017_06_14__09_33_43.tdms"
    original_time_range = [0.01150, 0.01162]  # Normal
    # original_time_range = [0.02667, 0.026730] # Defaut
    # original_time_range = [0.02783, 0.02790] # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)
    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    data__s_a_in = data__s_a_in[time_range]
    time__in = time__in[time_range]

    data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in,
                                                                                             parameter_file=parameter_file,
                                                                                             StartTime=
                                                                                             original_time_range[0],
                                                                                             return_processing=True)

    fig = plt.figure(figsize=(8, 2))
    prairie.use()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    ax1.plot(data_x, data_y)
    ax1.set_title('Optical position sensor A signal', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized amplitude')
    ax1.legend(['Signal (Sensor A)'], loc=1)
    prairie.style(ax1)
    plt.tight_layout()

    name = 'normal'
    plt.savefig("C:\Rapport_images_eps/" + name + '_OPS_signal.png', format='png', dpi=DPI)

    # original_time_range = [0.01150, 0.01162] # Normal
    original_time_range = [0.02667, 0.026730]  # Defaut
    # original_time_range = [0.02783, 0.02790] # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)
    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    data__s_a_in = data__s_a_in[time_range]
    time__in = time__in[time_range]

    data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in,
                                                                                             parameter_file=parameter_file,
                                                                                             StartTime=
                                                                                             original_time_range[0],
                                                                                             return_processing=True)

    fig = plt.figure(figsize=(8, 2))
    prairie.use()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    ax1.plot(data_x, data_y)
    ax1.set_title('Optical position sensor A signal', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized amplitude')
    ax1.legend(['Signal (Sensor A)'], loc=1)
    prairie.style(ax1)
    plt.tight_layout()

    name = 'default'
    plt.savefig("C:\Rapport_images_eps/" + name + '_OPS_signal.png', format='png', dpi=DPI)

    # original_time_range = [0.01150, 0.01162] # Normal
    # original_time_range = [0.02667, 0.026730] # Defaut
    original_time_range = [0.02783, 0.02790]  # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)
    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    data__s_a_in = data__s_a_in[time_range]
    time__in = time__in[time_range]

    data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in,
                                                                                             parameter_file=parameter_file,
                                                                                             StartTime=
                                                                                             original_time_range[0],
                                                                                             return_processing=True)

    fig = plt.figure(figsize=(8, 2))
    prairie.use()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    ax1.plot(data_x, data_y)
    ax1.set_title('Optical position sensor A signal', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized amplitude')
    ax1.legend(['Signal (Sensor A)'], loc=1)
    prairie.style(ax1)
    plt.tight_layout()

    name = 'reference'
    plt.savefig("C:\Rapport_images_eps/" + name + '_OPS_signal.png', format='png', dpi=DPI)


def camel_backs():
    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    file = "E:\BWS calibration (raw TDMS)\LAB BWS\PSB133rs_Offset06__2017_04_21__10_57\PSB3__P-2.00__S2____2017_04_21__11_20_14.tdms"
    original_time_range = [0.0147312, 0.0147594]  # Normal
    # original_time_range = [0.02667, 0.026730] # Defaut
    # original_time_range = [0.02783, 0.02790] # Reference
    # original_time_range = [0.009, 0.04] # All



    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)
    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)


    threshold_reference = np.amax(data__s_a_in) - data__s_a_in * np.mean(data__s_a_in)

    data__s_a_in = data__s_a_in[time_range]
    time__in = time__in[time_range]

    data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in,
                                                                                             parameter_file=parameter_file,
                                                                                             StartTime=
                                                                                             original_time_range[0],
                                                                                             return_processing=True,
                                                                                             camelback_threshold_on=False)

    fig = plt.figure(figsize=(8, 2.5))
    prairie.use()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    ax1.axhspan(0, bin, color='black', alpha=0.05)
    ax1.plot(data_x, data_y)
    ax1.plot(pck_x, pck_y, '.')
    ax1.plot(dwn_x, dwn_y, '.')
    ax1.set_title('OPS processing without camel-back correction', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized amplitude')
    ax1.legend(['Signal (Sensor A)', 'Detected peaks', 'Detected Valleys', 'Cleaning range'], loc=1)
    prairie.style(ax1)

    data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in,
                                                                                             parameter_file=parameter_file,
                                                                                             StartTime=
                                                                                             original_time_range[0],
                                                                                             return_processing=True)

    ax2.axhspan(0, bin, color='black', alpha=0.05)
    ax2.plot(data_x, data_y)
    ax2.plot(pck_x, pck_y, '.')
    ax2.plot(dwn_x, dwn_y, '.')
    ax2.set_title('Same signal with camel-back correction', loc='left')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Normalized amplitude')
    ax2.legend(['Signal (Sensor A)', 'Detected peaks', 'Detected Valleys', 'Cleaning range'], loc=1)
    prairie.style(ax2)

    name = 'camel-back_example'

    plt.tight_layout()

    plt.savefig('C:\Rapport_images_eps/' + name + '.png', format='png', dpi=DPI)


def algorithm_study():
    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    file = "E:\BWS calibration (raw TDMS)\LAB BWS\PSB133rs_Offset06__2017_04_21__10_57\PSB3__P-2.00__S2____2017_04_21__11_20_14.tdms"
    original_time_range = [0, 0.05]  # Normal
    # original_time_range = [0.02667, 0.026730] # Defaut
    # original_time_range = [0.02783, 0.02790] # Reference
    # original_time_range = [0.009, 0.04] # All

    config = configparser.RawConfigParser()
    config.read(parameter_file)
    SlitsperTurn = eval(config.get('OPS processing parameters', 'slits_per_turn'))
    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))
    rdcp = eval(config.get('OPS processing parameters', 'relative_distance_correction_prameters'))
    prominence = eval(config.get('OPS processing parameters', 'prominence'))
    camelback_threshold = eval(config.get('OPS processing parameters', 'camelback_threshold'))
    OPS_processing_filter_freq = eval(config.get('OPS processing parameters', 'OPS_processing_filter_freq'))

    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)
    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in,
                                                                                             parameter_file=parameter_file,
                                                                                             StartTime=0,
                                                                                             return_processing=True,
                                                                                             camelback_threshold_on=False)

    threshold_reference = np.amax(data__s_a_in) - data__s_a_in * np.mean(data__s_a_in)

    data__s_a_in = data__s_a_in[time_range]
    time__in = time__in[time_range]

    data__s_a_in_f = utils.butter_lowpass_filter(data__s_a_in, OPS_processing_filter_freq, sampling_frequency, order=5)

    max_data = np.amax(data__s_a_in)
    min_data = np.amin(data__s_a_in)

    max_data_f = np.amax(data__s_a_in_f)
    min_data_f = np.amin(data__s_a_in_f)

    data__s_a_in = data__s_a_in - min_data
    data__s_a_in = data__s_a_in / max_data

    data__s_a_in_f = data__s_a_in_f - min_data_f
    data__s_a_in_f = data__s_a_in_f / max_data_f

    data__s_a_in_f[np.where(data__s_a_in_f > 0.1)] = []
    data__s_a_in_f = utils.butter_lowpass_filter(data__s_a_in_f, OPS_processing_filter_freq, sampling_frequency,
                                                 order=5)

    maxtab, mintab = utils.peakdet(data__s_a_in, prominence)
    maxtab_f, mintab_f = utils.peakdet(data__s_a_in_f, prominence)

    locs_up = np.array(maxtab)[:, 0].astype(int)
    pck_up = np.array(maxtab)[:, 1]

    locs_dwn = np.array(mintab)[:, 0].astype(int)
    pck_dwn = np.array(mintab)[:, 1]

    locs_up_f = np.array(maxtab_f)[:, 0].astype(int)
    pck_up_f = np.array(maxtab_f)[:, 1]

    locs_dwn_f = np.array(mintab_f)[:, 0].astype(int)
    pck_dwn_f = np.array(mintab_f)[:, 1]

    fig = plt.figure(figsize=(8, 2.5))
    prairie.use()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    color1 = '#a6bddb'
    color2 = '#67a9cf'

    ax1.plot(time__in, data__s_a_in, color=color1, label='_nolegend_')
    ax1.plot(time__in, data__s_a_in_f, color=color2, label='_nolegend_')
    ax1.plot(time__in[locs_up], pck_up, '.', color=color1, markersize=8)
    ax1.plot(time__in[locs_up_f], pck_up_f, '.', color=color2, markersize=8)
    ax1.plot(1e-3 * thres_x, thres_y, '.k')
    # ax1.plot(data_x, data_y, '.', label='_nolegend_')
    # ax1.plot(pck_x, pck_y, '.', markersize=10)
    # ax1.plot(dwn_x, dwn_y, '.', markersize=10)
    # ax1.plot(thres_x, thres_y, '.k')
    # # ax1.plot(thres_x, thres_y, 'k')
    # ax1.plot([dwn_x[1], pck_x[2]], [dwn_y[1], pck_y[2]], alpha=0.2)
    ax1.set_title('Reflections detection strategies', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized amplitude')
    ax1.legend(['Maxima detection based on raw signal', 'Maxima detection based on filtered signal',
                'Middle rising-edge detection based on filtered signal'], loc=1)
    ax1.set_ylim([0., 0.65])
    ax1.set_xlim([0.0199896, 0.0200005])
    prairie.style(ax1)

    name = 'reflections_detection_strategies'

    Distances = np.diff(locs_up[0:locs_up.size - 1])
    RelDistr = np.divide(locs_up[1:locs_up.size], locs_up[0:locs_up.size - 1])

    Distances_f = np.diff(locs_up_f[0:locs_up_f.size - 1])
    RelDistr_f = np.divide(locs_up_f[1:locs_up_f.size], locs_up_f[0:locs_up_f.size - 1])

    Distances_ref = np.diff(thres_x[0:thres_x.size - 1])
    RelDistr_ref = np.divide(thres_x[1:thres_x.size], thres_x[0:thres_x.size - 1])

    print(np.std(RelDistr))
    print(np.std(RelDistr_f))
    print(np.std(RelDistr_ref))

    plt.tight_layout()

    plt.savefig('C:\Rapport_images_eps/' + name + '.png', format='png', dpi=DPI)


def single_reflection_study():
    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))
    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    file = "E:\BWS_Fatigue_Tests\Calibration_Bench_Files\\133rs_CC__2017_06_14__09_30\CALI__P37.00__S1____2017_06_14__09_33_43.tdms"
    original_time_range = [0.0115027, 0.0115065]  # Normal
    # original_time_range = [0.02667, 0.026730] # Defaut
    # original_time_range = [0.02783, 0.02790] # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)
    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    data__s_a_in = data__s_a_in[time_range]
    data__s_a_in = data__s_a_in - np.min(data__s_a_in)
    time__in = time__in[time_range]

    data__s_a_in_f = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    # data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in, parameter_file=parameter_file, StartTime=original_time_range[0], return_processing=True)

    fig = plt.figure(figsize=(8, 3))
    prairie.use()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    # ax2 = plt.subplot2grid((1,2), (0,1))


    (mu, sigma) = norm.fit(data__s_a_in)
    bins = np.arange(original_time_range[0], original_time_range[1],
                     (original_time_range[1] - original_time_range[0]) / 5000)
    y = mlab.normpdf(bins, 0.0115047, 0.0000020)

    def gaus(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    maxtab, mintab = utils.peakdet(data__s_a_in_f, 2000)
    locs_up_f = np.array(maxtab)[:, 0].astype(int)
    pck_up_f = np.array(maxtab)[:, 1]

    maxtab, mintab = utils.peakdet(data__s_a_in, 2000)
    locs_up = np.array(maxtab)[:, 0].astype(int)
    pck_up = np.array(maxtab)[:, 1]

    color1 = '#3690c0'
    color2 = '#016c59'

    popt, pcov = curve_fit(gaus, time__in, data__s_a_in, p0=[15000, 0.0115047, 0.0000020])

    ax1.plot(1e3 * time__in, data__s_a_in, linewidth=2, alpha=0.7, color=color1, label='_nolegend_')
    ax1.plot(1e3 * time__in[locs_up], pck_up, marker=7, markersize=10, color=color1)
    ax1.plot(1e3 * time__in, data__s_a_in_f, linewidth=2, alpha=0.7, color=color2, label='_nolegend_')
    ax1.plot(1e3 * time__in[locs_up_f], pck_up_f, marker=7, markersize=10, color=color2)
    ax1.plot(1e3 * time__in, gaus(time__in, *popt), '-k', linewidth=2, alpha=0.7, label='_nolegend_')
    ax1.plot(1e3 * popt[1], gaus(popt[1], *popt), marker=7, markersize=10, color='k')
    # ax1.plot(data_x, data_y)
    # ax1.plot(pck_x, pck_y, '.')
    # ax1.plot(dwn_x, dwn_y, '.')
    # ax1.plot(thres_x, thres_y, 'k')
    ax1.set_title('Optical position sensor A processing', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized amplitude')
    ax1.legend(['Raw signal and its maximum', 'Filtered signal and its maximum', 'Gaussian fit and its maximum'], loc=1)
    ax1.set_ylim([-100, 12000])
    prairie.style(ax1)

    # Data = ops.process_position(data__s_a_in, parameter_file=parameter_file, StartTime=original_time_range[0])
    #
    # time_SA = Data[0]
    # offset = 0
    #
    # distances_A = np.diff(time_SA)[offset:time_SA.size - 1 - offset]
    # rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
    #
    # ax2.plot(1e3 * time_SA[offset:time_SA.size - 2 - offset], rel_distances_A, '.', color='#018BCF')
    # ax2.set_xlabel('Time (ms)')
    # ax2.set_ylabel('Relative distance')
    # ax2.set_title('Error compensation', loc='left')
    # ax2.set_ylim([-0.5, 3])
    # dt.make_it_nice(ax2)
    plt.tight_layout()

    name = 'single_reflection_study'

    plt.savefig("C:\Rapport_images_eps/" + name + '.png', format='png', dpi=DPI)


def laser_study():
    parameter_file = '../data/parameters.cfg'
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    config.set('LabView output', 'data_SA', '[\'DISC PH. HOME dir IN\', \'DISC PH. HOME dir\']\'')
    config.set('LabView output', 'data_SB', '[\'DISC PH. IN dir IN\', \'DISC PH. IN dir HOME\']')
    config.set('LabView output', 'data_PD', '[\'WIRE PH. dir IN\', \'WIRE PH. dir HOME\']')
    #
    #
    file = "E:\BWS_Fatigue_Tests\Calibration_Bench_Files\\133rs_CC__2017_06_02__17_37\CALI__P95.00__S2____2017_06_02__17_53_59.tdms"

    original_time_range = [0.024, 0.0246]  # Normal
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    original_time_range = [0.016, 0.034]  # Normal
    time_range2 = np.arange(original_time_range[0] * sampling_frequency,
                            original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)

    data__p_d_in_f = utils.butter_lowpass_filter(data__p_d_in, 50e3, sampling_frequency)

    fig = plt.figure(figsize=(8, 2))
    prairie.use()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    ax1.plot(time__in[time_range2], 1e-3 * data__p_d_in[time_range2], linewidth=.5, color='#154E73')
    ax1.plot(time__in[time_range2], 1e-3 * data__p_d_in_f[time_range2], linewidth=1, color='black')
    ax1.set_title('Photodiode signal - Red laser', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('a.u.')
    prairie.style(ax1)

    ax2.plot(time__in[time_range], 1e-3 * data__p_d_in[time_range], linewidth=.5, color='#154E73')
    ax2.plot(time__in[time_range], 1e-3 * data__p_d_in_f[time_range], linewidth=1, color='black')
    ax2.set_title('Photodiode signal - Red laser - zoom', loc='left')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('a.u.')
    ax2.legend(['Raw signal', 'Filtered signal'])
    prairie.style(ax2)

    plt.tight_layout()
    plt.plot(block=False)

    config.set('LabView output', 'data_SA', 'DISC PHOTODIODE HOME')
    config.set('LabView output', 'data_SB', 'DISC PHOTODIODE IN')
    config.set('LabView output', 'data_PD', 'WIRE PHOTODIODE')

    name = 'red_laser_signal'

    plt.savefig("C:/Rapport_images_eps/" + name + '.png', format='png', dpi=DPI)

    # name1 = 'red_laser_signal'
    #
    # file = "E:\BWS calibration (raw TDMS)\LAB BWS\PSB133rs_Offset06_FixedTable_ShorterRail_300Scans_Center__2017_04_26__16_50\PSB3__P8.00__S21____2017_04_26__16_53_49.tdms"
    #
    # original_time_range = [0.02337, 0.02385] # Normal
    # time_range = np.arange(original_time_range[0]*sampling_frequency, original_time_range[1]*sampling_frequency).astype(int)
    #
    # original_time_range = [0.016, 0.034] # Normal
    # time_range2 = np.arange(original_time_range[0]*sampling_frequency, original_time_range[1]*sampling_frequency).astype(int)
    #
    # data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(file)
    #
    # data__p_d_in_f = utils.butter_lowpass_filter(data__p_d_in, 50e3, sampling_frequency)
    #
    #
    # fig = plt.figure(figsize=(8, 2))
    # prairie.use()
    # ax1 = plt.subplot2grid((1, 2), (0, 0))
    # ax2 = plt.subplot2grid((1, 2), (0, 1))
    #
    # ax1.plot(time__in[time_range2],1e-3*data__p_d_in[time_range2], linewidth=.5, color='#74A7EB')
    # ax1.plot(time__in[time_range2],1e-3*data__p_d_in_f[time_range2], linewidth=1, color='black')
    # ax1.set_title('Photodiode signal - Green laser', loc='left')
    # ax1.set_xlabel('Time (ms)')
    # ax1.set_ylabel('a.u.')
    # prairie.style(ax1)
    #
    # ax2.plot(time__in[time_range], 1e-3*data__p_d_in[time_range], linewidth=.5, color='#74A7EB')
    # ax2.plot(time__in[time_range], 1e-3*data__p_d_in_f[time_range], linewidth=1, color='black')
    # ax2.set_title('Photodiode signal - Green laser - zoom', loc='left')
    # ax2.set_xlabel('Time (ms)')
    # ax2.set_ylabel('a.u.')
    # ax2.legend(['Raw signal', 'Filtered signal'])
    # prairie.style(ax2)
    #
    # plt.tight_layout()
    #
    # name = 'green_laser_signal'
    #
    # plt.savefig("C:/Rapport_images_eps/" + name + '.png', format='png', dpi=DPI)
    #
    # # # plt.plot(block=True)
    # # pylab.show()


def laser_simulations(save=True, saving_name='laser_snr_vs_error_in_position'):
    data = sio.loadmat("F:\pyBWS01\data - Copy\\133rs_CC__2017_06_06__10_57 PROCESSED\PROCESSED_IN.mat")
    Data_SA = [data['time_SA'][0][0].squeeze(), data['angular_position_SA'][0][0].squeeze()]  # Get position data from a PROCESSED file

    SNR = np.arange(2, 14, 0.5)
    x = np.arange(0.0245, 0.0249, 1 / 20e6)

    # Define double laser signal characteristics based on experimental data
    mu1 = 0.0245881
    mu2 = 0.0248099
    sigma1 = 6e-6
    sigma2 = sigma1

    As = 20000
    f = 120e3

    G1 = mlab.normpdf(x, mu1, sigma1)
    G1 = G1 / np.max(G1) * As
    G2 = mlab.normpdf(x, mu2, sigma2)
    G2 = G2 / np.max(G2) * As

    fig = plt.figure(figsize=(10, 3))
    prairie.use()
    ax1 = fig.add_subplot(111)

    Occlusion = []

    for snr in SNR:

        temp = []

        G1 = mlab.normpdf(x, mu1, sigma1)
        G1 = G1 / np.max(G1) * As
        G2 = mlab.normpdf(x, mu2, sigma2)
        G2 = G2 / np.max(G2) * As

        for i in np.arange(0, 500, 1):
            phi = random.random() * np.pi * 2
            sM = As / snr * np.sin(2 * np.pi * f * x + phi)
            signal = -(sM + G1 + G2)

            occlusions = ops.find_occlusions(signal)

            # ax1.plot(x, signal)
            # ax1.plot(x[occlusions], signal[occlusions], '.k')

            Data_SA_R = utils.resample(Data_SA, np.array([x.squeeze(), signal.squeeze()]))
            occ1 = Data_SA_R[1][int(occlusions[0])]
            occ2 = Data_SA_R[1][int(occlusions[1])]
            _occlusion = (occ2 - occ1) / 2 + occ1

            temp.append(occ1)

        Occlusion.append(np.std(np.asarray(temp)) * 150e-3 * 1e6)
        # Occlusion.append(np.std(np.asarray(temp)))

    ax1.plot(SNR, Occlusion, '.k')
    ax1.set_title('Influence of the double laser SNR on the measured wire position', loc='left')
    ax1.set_xlabel('Double laser signal SNR')
    ax1.set_ylabel('Induced error in projected wire position (\u03BCm)')
    plt.tight_layout()

    prairie.style(ax1)
    plt.savefig("C:/Rapport_images_eps/" + 'laser_snr_vs_error_in_position' + '.png', format='png', dpi=DPI)

    G1 = mlab.normpdf(x, mu1, sigma1)
    G1 = G1 / np.max(G1) * As
    G2 = mlab.normpdf(x, mu2, sigma2)
    G2 = G2 / np.max(G2) * As

    sM = As / 4 * np.sin(2 * np.pi * f * x)
    signal = -(sM + G1 + G2)

    fig2 = plt.figure(figsize=(10, 3))
    prairie.use()
    # plt.rc('text', usetex=True)
    ax2 = fig2.add_subplot(121)
    ax3 = fig2.add_subplot(122)

    ax2.plot(1e3 * x[0:G1.size / 2], -1e-3 * G1[0:G1.size / 2], linewidth=4, alpha=0.5,
             label=r"$\mathcal{A}_S * \mathcal{G}_1$")
    ax2.plot(1e3 * x[G2.size / 2::], -1e-3 * G2[G2.size / 2::], linewidth=4, alpha=0.5,
             label=r"$\mathcal{A}_S * \mathcal{G}_2$")
    ax2.plot(1e3 * x, 1e-3 * signal,
             label=r" signal = $\mathcal{A}_S \times (\mathcal{G}_1 + \mathcal{G}_2) + \mathcal{A}_M \times \sin(x + \phi)$")
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('a.u.')
    ax2.set_title('Double laser signal reconstruction (example with SNR=4)', loc='left')
    ax2.set_ylim([-1e-3 * 30000, 1e-3 * 23000])

    ax3.plot(1e3 * x, 1e-3 * signal, label='signal')
    ax3.plot(1e3 * x, 1e-3 * utils.butter_lowpass_filter(signal, 1e5, 20e6, order=5), linewidth=4, alpha=0.5,
             label='filtered signal')
    occlusions = ops.find_occlusions(signal)
    ax3.plot(1e3 * x[occlusions], 1e-3 * signal[occlusions], '.k', label='detected peaks')
    ax3.legend(loc='upper right')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('a.u.')
    ax3.set_title('Occlusions detection', loc='left')
    ax3.set_ylim([-1e-3 * 30000, 1e-3 * 23000])

    prairie.style(ax2)
    prairie.style(ax3)
    plt.tight_layout()

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def errors():
    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    fig = plt.figure(figsize=(9, 9))
    prairie.use()

    PROCESSED_folder = "F:\pyBWS01\data\\133rs_CC__2017_06_01__10_45 PROCESSED"

    data = sio.loadmat(PROCESSED_folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)

    config.set('LabView output', 'data_SA', '[\'DISC PH. HOME dir IN\', \'DISC PH. HOME dir\']\'')
    config.set('LabView output', 'data_SB', '[\'DISC PH. IN dir IN\', \'DISC PH. IN dir HOME\']')
    config.set('LabView output', 'data_PD', '[\'WIRE PH. dir IN\', \'WIRE PH. dir HOME\']')

    original_time_range = [0.02645, 0.0271]  # Reference
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    time_SA = data['time_SA']
    offset = 0
    ax1 = plt.subplot2grid((3, 3), (0, 2))

    for time in time_SA:
        distances_A = np.diff(time)[offset:time.size - 1 - offset]
        rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
        ax1.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')

    ax1.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax1.set_title('LAB Prototype - 133rs - 01.06.17', loc='left')
    ax1.set_ylim([-0.5, 3])
    ax1.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax1)

    PROCESSED_folder = "F:/pyBWS01/data/133rs_CC__2017_06_15__09_26 PROCESSED"

    config.set('LabView output', 'data_SA', 'DISC PHOTODIODE HOME')
    config.set('LabView output', 'data_SB', 'DISC PHOTODIODE IN')
    config.set('LabView output', 'data_PD', 'WIRE PHOTODIODE')

    data = sio.loadmat(PROCESSED_folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)

    original_time_range = [0.0265, 0.0271]  # Reference
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    time_SA = data['time_SA']
    offset = 0

    ax2 = plt.subplot2grid((3, 3), (0, 1))
    ax3 = plt.subplot2grid((3, 3), (0, 0))

    for time in time_SA:
        distances_A = np.diff(time)[offset:time.size - 1 - offset]
        rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
        ax2.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')
        ax3.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')

    ax2.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax2.set_title('LAB Prototype - 133rs - 15.06.17', loc='left')
    ax2.set_ylim([-0.5, 3])
    ax2.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax2)

    original_time_range = [0.0311, 0.0322]  # Reference

    ax3.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax3.set_title('LAB Prototype - 133rs - 15.06.17', loc='left')
    ax3.set_ylim([-0.5, 3])
    ax3.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax3)

    PROCESSED_folder = "F:/PSB133rs__2017_03_10__19_27 PROCESSED"

    config.set('LabView output', 'data_SA', 'DISC PHOTODIODE HOME')
    config.set('LabView output', 'data_SB', 'DISC PHOTODIODE IN')
    config.set('LabView output', 'data_PD', 'WIRE PHOTODIODE')

    data = sio.loadmat(PROCESSED_folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)

    original_time_range = [0.0124, 0.01272]  # Reference
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    time_SA = data['time_SA']
    offset = 0

    ax4 = plt.subplot2grid((3, 3), (1, 0))

    for time in time_SA:
        distances_A = np.diff(time)[offset:time.size - 1 - offset]
        rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
        ax4.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')

    ax4.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax4.set_ylabel('Relative distance')
    ax4.set_title('PSB Prototype - 133rs - 10.03.17', loc='left')
    ax4.set_ylim([-0.5, 3])
    ax4.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax4)

    PROCESSED_folder = "F:/pyBWS01/data/Not_Vacuum_55rs_Calibration__2017_05_22__11_59 PROCESSED"

    config.set('LabView output', 'data_SA', 'DISC PHOTODIODE HOME')
    config.set('LabView output', 'data_SB', 'DISC PHOTODIODE IN')
    config.set('LabView output', 'data_PD', 'WIRE PHOTODIODE')

    data = sio.loadmat(PROCESSED_folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)

    original_time_range = [0.0510, 0.0517]  # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    time_SA = data['time_SA']
    offset = 0

    # fig = plt.figure(figsize=(15, 3))
    # dt.nice_style()
    ax5 = plt.subplot2grid((3, 3), (1, 1))

    for time in time_SA:
        distances_A = np.diff(time)[offset:time.size - 1 - offset]
        rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
        ax5.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')

    ax5.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax5.set_title('PSB Prototype - 133rs - 22.05.17', loc='left')
    ax5.set_ylim([-0.5, 3])
    ax5.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax5)

    PROCESSED_folder = "F:\pyBWS01\data\Vacum_Calibration_55rs__2017_05_18__15_22 PROCESSED"

    config.set('LabView output', 'data_SA', 'DISC PHOTODIODE HOME')
    config.set('LabView output', 'data_SB', 'DISC PHOTODIODE IN')
    config.set('LabView output', 'data_PD', 'WIRE PHOTODIODE')

    data = sio.loadmat(PROCESSED_folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)

    original_time_range = [0.0519, 0.05258]  # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    time_SA = data['time_SA']
    offset = 0

    # fig = plt.figure(figsize=(15, 3))
    # dt.nice_style()
    ax6 = plt.subplot2grid((3, 3), (1, 2))
    ax7 = plt.subplot2grid((3, 3), (2, 0))

    for time in time_SA:
        distances_A = np.diff(time)[offset:time.size - 1 - offset]
        rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
        ax6.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')
        ax7.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')

    ax6.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax6.set_title('PSB Prototype - 55rs - 18.05.17', loc='left')
    ax6.set_ylim([-0.5, 3])
    ax6.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax6)

    original_time_range = [0.0622, 0.06275]  # Reference

    ax7.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax7.set_title('PSB Prototype - 55rs - 18.05.17', loc='left')
    ax7.set_ylim([-0.5, 3])
    ax7.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax7)

    PROCESSED_folder = "F:\pyBWS01\data\\133rs_CC__2017_06_15__09_26 PROCESSED"

    config.set('LabView output', 'data_SA', '[\'DISC PH. HOME dir IN\', \'DISC PH. HOME dir\']\'')
    config.set('LabView output', 'data_SB', '[\'DISC PH. IN dir IN\', \'DISC PH. IN dir HOME\']')
    config.set('LabView output', 'data_PD', '[\'WIRE PH. dir IN\', \'WIRE PH. dir HOME\']')

    data = sio.loadmat(PROCESSED_folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)

    original_time_range = [0.02645, 0.0275]  # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    time_SA = data['time_SA']
    offset = 0

    # fig = plt.figure(figsize=(15, 3))
    # dt.nice_style()
    ax8 = plt.subplot2grid((3, 3), (2, 1))
    ax9 = plt.subplot2grid((3, 3), (2, 2))

    for time in time_SA:
        distances_A = np.diff(time)[offset:time.size - 1 - offset]
        rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
        ax8.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')
        ax9.plot(1e3 * time[offset:time.size - 2 - offset], rel_distances_A, '.')

    ax8.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax8.set_xlabel('Time (ms)')
    ax8.set_title('LAB Prototype - 55rs - 15.06.17', loc='left')
    ax8.set_ylim([-0.5, 3])
    ax8.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax8)

    original_time_range = [0.030, 0.032]  # Reference

    ax9.axhspan(1.8, 2.1, color='black', alpha=0.1)
    ax9.set_title('LAB Prototype - 133rs - 15.06.17', loc='left')
    ax9.set_ylim([-0.5, 3])
    ax9.set_xlim(np.asarray(original_time_range) * 1e3)
    prairie.style(ax9)

    plt.tight_layout()
    #
    # pylab.show()

    name = 'multiple_defaults'

    plt.savefig("G:/Users/l/ligarcia/Desktop\Rapport de Stage\Images\OPS processing/" + name + '.png', format='png',
                dpi=DPI)


def signal_error(case=0):
    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    file = "E:\BWS_Fatigue_Tests\Calibration_Bench_Files\\133rs_CC__2017_06_14__09_30\CALI__P37.00__S1____2017_06_14__09_33_43.tdms"
    if case == 0:
        original_time_range = [0.01150, 0.01162]  # Normal
        name = 'REMP_normal_signal'
    elif case == 1:
        original_time_range = [0.02667, 0.026730] # Defaut
        name = 'REMP_default_signal'
    elif case == 2:
        original_time_range = [0.02783, 0.02790] # Reference
        name = 'REMP_reference_signal'
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
        file)
    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    data__s_a_in = data__s_a_in[time_range]
    time__in = time__in[time_range]

    data_x, data_y, pck_x, pck_y, dwn_x, dwn_y, thres_x, thres_y, bin = ops.process_position(data__s_a_in,
                                                                                             parameter_file=parameter_file,
                                                                                             StartTime=
                                                                                             original_time_range[0],
                                                                                             return_processing=True)

    fig = plt.figure(figsize=(10, 3))
    prairie.use()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    ax1.plot(data_x, data_y)
    ax1.plot(pck_x, pck_y, '.')
    ax1.plot(dwn_x, dwn_y, '.')
    ax1.plot(thres_x, thres_y, '.-', color='k')
    ax1.set_title('Optical position sensor A processing', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Normalized amplitude')
    ax1.legend(['Signal (Sensor A)', 'Detected peaks', 'Detected Valleys', 'REMP'], loc=1)
    prairie.style(ax1)

    Data = ops.process_position(data__s_a_in, parameter_file=parameter_file, StartTime=original_time_range[0])

    time_SA = Data[0]
    offset = 0

    distances_A = np.diff(time_SA)[offset:time_SA.size - 1 - offset]
    rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])

    ax2.plot(1e3 * time_SA[offset:time_SA.size - 2 - offset], rel_distances_A, '.', color='#018BCF')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Relative distance')
    ax2.set_title('RDS plot', loc='left')
    ax2.set_ylim([-0.5, 3])
    prairie.style(ax2)
    plt.tight_layout()


    plt.savefig("C:/Rapport_images_eps/" + name + '.png', format='png', dpi=DPI)


def beam_displacer():

    wavelengths = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2,
                   1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2, 2.05,
                   2.1, 2.15, 2.2, 2.25, 2.3]

    displacement =[3.272583584, 3.109208456, 3.031873705, 2.98201882, 2.945326613, 2.916664823, 2.893462941,
                   2.874141822, 2.857633235, 2.843182094, 2.830241924, 2.81841079, 2.807389066, 2.796950633,
                   2.786922848, 2.777172413, 2.767595306, 2.758109514, 2.748649743, 2.739163525, 2.729608319,
                   2.719949341, 2.71015792, 2.700210234, 2.690086344, 2.679769441, 2.669245251, 2.658501573,
                   2.647527905, 2.63631515, 2.624855375, 2.61314162, 2.601167737, 2.588928262, 2.57641831,
                   2.563633482, 2.550569796, 2.537223626, 2.523591644, 2.509670785]

    fig = plt.figure(figsize=(5, 3))
    prairie.use()
    ax1 = fig.add_subplot(111)

    ax1.plot(wavelengths, displacement)

    ax1.set_xlabel('\u03BB (\u03BCm)')
    ax1.set_ylabel('Beam separation distance (mm)')
    ax1.set_title('Calcite Beam Displacers separation vs wavelength \u03BB', loc='left')

    prairie.style(ax1)

    plt.tight_layout()

    plt.savefig("C:/Rapport_images_eps/" + 'beam_displacement_THORLABS' + '.png', format='png', dpi=DPI)


def vacuum_test(folders, save=False, saving_name=None):
    fig = plt.figure(figsize=(8, 2.5))
    prairie.use()

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    M = []

    i = 0

    for folder in folders:
        data = sio.loadmat(folder + '/calibration_results.mat')
        residuals = data['residuals_IN_origin_mean'][0]
        laser_position = data['laser_position_IN_mean'][0]
        if i == 0:
            ax1.plot(laser_position, utils.butter_lowpass_filter(residuals, 1 / 101, 1 / 10) - np.mean(residuals),
                     alpha=0.2, linewidth=2,
                     label='Calibration residuals profiles (' + str(len(folders)) + ' - filtered and centered)')
        else:
            ax1.plot(laser_position, utils.butter_lowpass_filter(residuals, 1 / 101, 1 / 10) - np.mean(residuals),
                     alpha=0.2, linewidth=2, label='_nolegend_')
        M.append(residuals)
        i += 1

    M = np.asarray(M)
    M = np.mean(M, 0)

    ax1.plot(laser_position, utils.butter_lowpass_filter(M, 1 / 101, 1 / 10), color='k', linewidth=2.5,
            label='Mean residual profile')
    ax1.set_xlabel('Laser position (mm)')
    ax1.set_ylabel('Residual error (\u03BCm)')
    ax1.legend()

    plt.tight_layout()
    prairie.style(ax1)

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def theoretical_movement(save=False, saving_name=None):

    range = np.arange(34.6, 34.6+180, 0.1)*np.pi/180

    fig = plt.figure(figsize=(5, 3))
    prairie.use()
    ax = fig.add_subplot(111)

    ax.plot(range, 150 + 85*np.cos(range))
    ax.set_xlabel('\u03B8 (radian)')
    ax.set_ylabel('y(\u03B8) (mm)')
    ax.set_title('Ideal mechanical movement of the wire', loc='left')

    plt.tight_layout()

    prairie.style(ax)

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def scan_outome():


    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    file = "E:\BWS calibration (raw TDMS)\LAB BWS\PSB133rs_Offset06__2017_04_21__10_57\PSB3__P0.00__S2____2017_04_21__11_20_56.tdms"
    original_time_range = [0.00, 0.499]  # Normal
    # original_time_range = [0.02667, 0.026730] # Defaut
    # original_time_range = [0.02783, 0.02790] # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    tdms_file = TdmsFile(file)

    data__s_a = tdms_file.object('Picoscope Data', 'DISC PHOTODIODE HOME').data
    data__s_b = tdms_file.object('Picoscope Data', 'DISC PHOTODIODE IN').data
    data__p_d = tdms_file.object('Picoscope Data', 'WIRE PHOTODIODE').data

    # data__s_a_in = utils.butter_lowpass_filter(data__s_a_in, 1e6, sampling_frequency, order=5)

    #data__s_a_in = data__s_a_in[time_range]
    #time__in = time__in[time_range]

    fig = plt.figure(figsize=(8, 2))
    prairie.use()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    ax1.plot(np.arange(0, data__s_a.size)/20e6, data__s_a)
    ax1.plot(np.arange(0, data__s_b.size)/20e6, data__s_b)
    ax1.set_title('OPS Signal from a complete scan ', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.legend(['Signal from OPS A', 'Signal from OPS B'], loc='lower right')
    prairie.style(ax1)
    plt.tight_layout()

    plt.savefig("C:\Rapport_images_eps/complete_OPS_signal.png", format='png', dpi=DPI)

    fig = plt.figure(figsize=(8, 2))
    prairie.use()
    ax1 = plt.subplot2grid((1, 2), (0, 0))

    ax1.plot(np.arange(0, data__p_d.size)/20e6, data__p_d)
    ax1.set_title('Photodiode Signal from a complete scan', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (a.u.)')
    prairie.style(ax1)
    plt.tight_layout()

    ax2 = plt.subplot2grid((1, 2), (0, 1))

    original_time_range = [0.0224, 0.024]  # Normal
    # original_time_range = [0.02667, 0.026730] # Defaut
    # original_time_range = [0.02783, 0.02790] # Reference
    # original_time_range = [0.009, 0.04] # All
    time_range = np.arange(original_time_range[0] * sampling_frequency,
                           original_time_range[1] * sampling_frequency).astype(int)

    data__p_d = data__p_d[time_range]

    ax2.plot(time_range/20e6, data__p_d)
    ax2.set_title('Photodiode signal when the wire cross the laser', loc='left')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (a.u.)')
    prairie.style(ax2)
    plt.tight_layout()

    plt.savefig("C:\Rapport_images_eps/complete_PD_signal.png", format='png', dpi=DPI)



