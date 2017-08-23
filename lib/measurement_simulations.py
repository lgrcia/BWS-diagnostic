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


import numpy as np
import scipy.io as sio
import configparser
import matplotlib.pyplot as plt


from tqdm import tqdm
from scipy.optimize import curve_fit

from lib import prairie
import lib.utils as utils

DPI = 200


def simulate_wire_profile_measurements(reference_fit_parameters_file, fit_parameters_file, gauss_beam_position=0,
                                       gauss_beam_sigma=0, diagnostic=False, save=False, saving_name=None, position_random_error=15e-6):

    ##########################
    # Loading files and data #
    ##########################

    param_file = sio.loadmat(fit_parameters_file + '/calibration_results.mat', struct_as_record=False, squeeze_me=True)
    fit_parameters = param_file['f_parameters_IN']

    param_file = sio.loadmat(reference_fit_parameters_file, struct_as_record=False, squeeze_me=True)
    reference_fit_parameters = param_file['f_parameters_IN']

    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    measurement_sigma = position_random_error

    # gauss_beam_position = eval(config.get('Beam characteristics', 'gauss_beam_position'))
    # gauss_beam_sigma = eval(config.get('Beam characteristics', 'gauss_beam_sigma'))

    scan_speed = eval(config.get('Simulation', 'scan_speed'))
    cycle_period = eval(config.get('Simulation', 'cycle_period'))

    point_per_sigma = gauss_beam_sigma / scan_speed / cycle_period

    #############################
    # Sigma and Mean extraction #
    #############################

    if diagnostic is True :
        fig = plt.figure(figsize=(8, 2.5))
        prairie.use()
        ax = fig.add_subplot(111)

    sigmas = []

    extended_std = 15 * gauss_beam_sigma
    beam_offset_center = utils.theoretical_laser_position(
        utils.inverse_theoretical_laser_position(gauss_beam_position, *reference_fit_parameters), *fit_parameters)

    ideal_beam_measurement_positions = np.arange(gauss_beam_position - extended_std / 2,
                                                 gauss_beam_position + extended_std / 2,
                                                 gauss_beam_sigma / point_per_sigma)
    
    disk_measurement_positions = utils.inverse_theoretical_laser_position(ideal_beam_measurement_positions,
                                                                          *reference_fit_parameters)
    beam_measurement_positions = utils.theoretical_laser_position(disk_measurement_positions, *fit_parameters)

    def gauss(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    for i in np.arange(0, 1000, 1):

        sim_measured_positions = []
        sim_amplitudes = []

        for mu in beam_measurement_positions:
            sim_pos = mu + np.random.normal(0, measurement_sigma)
            sim_measured_positions.append(sim_pos)
            sim_amp = gauss(mu, 1, beam_offset_center, gauss_beam_sigma)
            sim_amplitudes.append(sim_amp)

        x = sim_measured_positions
        y = sim_amplitudes

        parameters, trash = curve_fit(gauss, x, y, p0=[2, beam_offset_center, gauss_beam_sigma])

        if diagnostic is True:

            arange = np.arange(beam_measurement_positions[0],
                               beam_measurement_positions[::-1][0],
                               (beam_measurement_positions[1]-beam_measurement_positions[0])/100)
            ax.plot(1e3*arange, gauss(arange, *parameters), linewidth=0.2)
            ax.plot(1e3*np.asarray(x), y, '.k', markersize=4)

            scan_speed = 133 * 150e-3
            cycle_period = 20e-6

            pps = gauss_beam_sigma / scan_speed / cycle_period

            ax.set_title('Beam profile simulation' + ' [ \u03C3:' + "{:3.0f}".format(1e6*gauss_beam_sigma) + '\u03BCm  ' + '\u03BC:' + "{:3.0f}".format(1e6*gauss_beam_position) + ' \u03BCm ] - 1000 profiles\n' + "{:3.2f}".format(pps) + ' points per sigma', loc='left')
            ax.set_xlabel('Transverse dimension (mm)')
            ax.set_ylabel('Secondary shower intensity (a.u)')
            ax.legend(['Beam gaussian fits', 'Measurement points'])

        sigmas.append(parameters[2])

    if diagnostic is True:
        prairie.style(ax)
        plt.tight_layout()

        if save is True:
            plt.show(block=False)

            if saving_name is not None:
                plt.savefig(saving_name + '.png', format='png', dpi=DPI)
                plt.close('all')
            else:
                print('saving_name is None - Figure not saved')
        else:
            plt.show(block=True)

    return [np.mean(sigmas), np.std(sigmas), beam_offset_center]


def simulation_of_beam_width_measurements_error(reference_fit_parameters_file, fit_parameters_file, save=False, saving_name=None, position_random_error=15e-6, theoretical_curve=False):

    fig = plt.figure(figsize=(8, 2.5))
    prairie.use()
    ax = fig.add_subplot(111)

    legend = []

    for center in np.arange(0, 1, 1):

        r = []
        sigmas = []
        means = []

        sig = 150 * np.logspace(-6, -4, 20)

        for beam_sigma in tqdm(sig):
            r = simulate_wire_profile_measurements(reference_fit_parameters_file, fit_parameters_file, center, beam_sigma, position_random_error=position_random_error)
            sigmas.append(r[1]/np.sqrt(2))
            means.append(r[0])

        sigmas = np.asarray(sigmas)
        means = np.asarray(means)
        legend.append('center: ' + str(center) + ' mm')

        Error = 100 * sigmas / means
        Error = 100 * sigmas/sig

        scan_speed = 133 * 150e-3
        cycle_period = 20e-6

        pps = sig / scan_speed / cycle_period

        ax.loglog(pps, Error, '.-')

        if theoretical_curve is True:
            SNR_x = sig/sigmas
            theor_curve = 100 * 0.4/(SNR_x*np.sqrt(pps))
            ax.loglog(pps, theor_curve, '-')
            legend.append('Theoretical curve')

    # ax.set_title('Simulation of beam width measurements error (BWS LabProto2 ' + fit_parameters_file + ')', loc='left')
    ax.set_title('Relative random error on beam size measurements', loc='left')
    ax.set_xlabel('Points per sigma')
    ax.set_ylabel('Error (%)')
    ax.legend(legend)
    prairie.style(ax, ticks=None)
    plt.tight_layout()

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
            plt.close('all')
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def Beam_width_error_over_different_calibration_curves(reference_fit_parameters_file, folders, save=False, saving_name=None, position_random_error=15e-6):
    fig = plt.figure(figsize=(8, 2.5))
    prairie.use()
    ax = fig.add_subplot(111)
    legend = []

    sigmas = []
    means = []
    i = 0
    center = []

    beam_width = 150e-6

    scan_speed = 133 * 150e-3
    cycle_period = 20e-6
    pps = beam_width / scan_speed / cycle_period


    for folder in tqdm(folders):
        r = simulate_wire_profile_measurements(reference_fit_parameters_file, folder, 0, beam_width, position_random_error=position_random_error)
        sigmas.append(r[1])
        means.append(r[0])
        center.append(r[2])

        # ax.plot(i, 100*r[1]/r[0], '.')
        i += 1

    sigmas = np.asarray(sigmas)
    means = np.asarray(means)

    Error = 100 * sigmas / means
    Error = np.asarray(Error)

    ax.plot(np.asarray(Error), '.', color='k')
    ax.set_ylabel('beam width error (%)')
    ax.set_xlabel('Calibrations #')
    ax.set_ylim([np.mean(Error) - 6*np.std(Error), np.mean(Error) + 6*np.std(Error)])
    ax.set_title('Beam width error over different calibration curves \n (' + "{:3.2f}".format(pps) + ' points per sigma'')', loc='left')
    # ax.set_ylim([0.5, 1.5])
    prairie.style(ax)

    #
    # scan_speed = 133 * 150e-3
    # cycle_period = 20e-6
    #
    # pps = sig / scan_speed / cycle_period
    #
    # ax.loglog(pps, Error, '.-')
    #
    # ax.set_title('Simulation of beam width measurements error (BWS LabProto2)', loc='left')
    # ax.set_xlabel('Points per sigma')
    # ax.set_ylabel('Error (%)')
    # ax.legend(legend)
    # prairie.apply(ax, ticks=None)
    plt.tight_layout()

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
            plt.close('all')
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)
