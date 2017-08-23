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


import os
import random
import numpy as np
import configparser
import scipy.io as sio

from math import sqrt
from matplotlib import mlab
from nptdms import TdmsFile
from scipy.stats import norm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import patches as mpatches
from sklearn.metrics import mean_squared_error

from lib import utils
from lib import prairie
from lib import ops_processing as ops


DPI = 200


# Plotting tools ----------------------------------------------------------------------------------


def make_histogram(data, data_range, units, axe=None, color=None, projected=None, zorder=None):
    """
    Make an histogram of data inside a certain data_range and add a gaussian fit in the legend
    
    Args:
        data: data to be analyse
        data_range: data range to be histogramed
        units: units of the data
        axe: axe that will carry the plot
        color: color of the plot
         
        
    Returns: legend giving mean value and sigma of the gaussian fit
    """

    legends = []
    means = []
    sigmas = []

    if projected is True:
        parameter_file = utils.resource_path('data/parameters.cfg')
        config = configparser.RawConfigParser()
        config.read(parameter_file)
        fork_length = eval(config.get('Scanner parameters', 'fork_length'))


    if axe is None:

        data_range = np.asarray(data_range)
        plt.hist(data, 50, normed=1, alpha=0.75)
        plt.xlim([data_range[0], data_range[1]])
        (mu, sigma) = norm.fit(data)
        bins = np.arange(data_range[0], data_range[1], (data_range[1]-data_range[0])/500)
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, linewidth=2, color='black')
        plt.legend(['\u03C3 : ' + "{:3.3f}".format(sigma/np.sqrt(2)) + ' ' + units + '\n' + '\u03BC: ' + "{:3.3f}".format(mu) + ' ' + units])

    else:

        data_range = np.asarray(data_range)
        if color is not None:
            axe.hist(data, 50, normed=1, alpha=0.75, color=color)
        else:
            axe.hist(data, 50, normed=1, alpha=0.75)

        axe.set_xlim([data_range[0], data_range[1]])
        (mu, sigma) = norm.fit(data)
        # bins = np.arange(data_range[0], data_range[1], 0.1)
        bins = np.arange(data_range[0], data_range[1], (data_range[1] - data_range[0]) / 500)
        y = mlab.normpdf(bins, mu, sigma)
        if color is not None:
            if zorder is not None:
                axe.plot(bins, y, linewidth=2, color=color, zorder=zorder)
            else:
                axe.plot(bins, y, linewidth=2, color=color)
        else:
            axe.plot(bins, y, linewidth=2, color='k')

        if axe.get_legend() is not None:
            texts = axe.get_legend().get_texts()
            for text in texts:
                legends.append(text._text)

        if projected is None:
            legends.append('\u03C3 ' + "{:3.3f}".format(sigma / np.sqrt(2))+ '   ' + '\u03BC ' + "{:3.3f}".format(mu) + '  (' + units + ')')
        elif projected is True:
            legends.append('\u03C3 ' + "{:3.3f}".format(sigma / np.sqrt(2)) + '   ' + '\u03BC ' + "{:3.3f}".format(mu) + '  (' + units + ') \n' +
                           '\u03C3 ' + "{:3.3f}".format(sigma / np.sqrt(2) * fork_length) + '   ' + '\u03BC ' + "{:3.3f}".format(mu * fork_length) + '  (\u03BCm)')



        means.append(str(mu))
        sigmas.append(str(sigma / np.sqrt(2)))

        axe.legend(legends, loc='upper right')

    return ["{:3.2f}".format(mu), "{:3.2f}".format(sigma / np.sqrt(2))]


# Single calibration analysis  ---------------------------------------------------------------------


def plot_calibration(folder_name, in_or_out, save=False, saving_name=None, complete_residuals_curve=False, separate=False):
    """
    Plot the complete calibration figure of a measurement set

    Args:
        folder_name: folder containing PROCESSED_IN and PROCESSED_OUT, the processed data
        in_or_out: 'IN' or 'OUT' analysis
        save: save the figure as png (figure will not be displayed if True)
        saving_name: Name of the png file (without extension)
    """

    if in_or_out is 'IN':
        filename = 'PROCESSED_IN.mat'
        color = '#018BCF'
    elif in_or_out is 'OUT':
        filename = 'PROCESSED_OUT.mat'
        color = '#0EA318'

    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)
    positions_for_fit = eval(config.get('OPS processing parameters', 'positions_for_fit'))
    positions_for_analysis = eval(config.get('OPS processing parameters', 'positions_for_analysis'))
    tank_center = eval(config.get('OPS processing parameters', 'offset_center'))

    data = sio.loadmat(folder_name + '/' + filename, struct_as_record=False, squeeze_me=True)
    occlusion_position = data['occlusion_position']
    laser_position = data['laser_position']
    idxs = np.argsort(laser_position)
    occlusion_position = occlusion_position[idxs]
    laser_position = laser_position[idxs]

    laser_position = -laser_position + tank_center

    if in_or_out is 'OUT':
        occlusion_position = np.pi / 2 - occlusion_position

    unique_laser_position = np.unique(laser_position)
    occlusion_position_mean = []

    for laser_pos in unique_laser_position:
        occlusion_position_mean.append(np.mean(occlusion_position[np.where(laser_position == laser_pos)[0]]))

    off1 = [int(positions_for_fit[0] / 100 * unique_laser_position.size),
            int(positions_for_fit[1] / 100 * unique_laser_position.size)]

    occlusion_position_mean = np.asarray(occlusion_position_mean)
    popt, pcov = curve_fit(utils.theoretical_laser_position, occlusion_position_mean[off1[0]:off1[1]],
                           unique_laser_position[off1[0]:off1[1]], bounds=([-5, 80, 100], [5, 500, 500]))
    theorical_laser_position_mean = utils.theoretical_laser_position(occlusion_position_mean, popt[0], popt[1], popt[2])
    theoretical_laser_position = utils.theoretical_laser_position(occlusion_position, popt[0], popt[1], popt[2])
    param = popt

    off2 = [int(positions_for_analysis[0] / 100 * laser_position.size),
            int(positions_for_analysis[1] / 100 * laser_position.size)]

    laser_position = laser_position[off2[0]:off2[1]]
    theoretical_laser_position = theoretical_laser_position[off2[0]:off2[1]]
    occlusion_position = occlusion_position[off2[0]:off2[1]]
    residuals = laser_position - theoretical_laser_position

    if complete_residuals_curve is True:
        d = 3
    else:
        d = 2

    if separate is True:
        fig = plt.figure(figsize=(8, 2.5))
        prairie.use()
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax3 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig = plt.figure(figsize=(8, d*2))
        prairie.use()
        ax2 = plt.subplot2grid((d, 2), (1, 1))
        ax3 = plt.subplot2grid((d, 2), (1, 0))
        ax1 = plt.subplot2grid((d, 2), (0, 0), colspan=2)

    residuals = residuals[off2[0]:off2[1]]
    make_histogram(1e3*residuals, [-300, 300], '\u03BCm', axe=ax2, color=color)
    ax2.set_title('Wire position error histogram', loc='left')
    ax2.set_xlabel('Wire position error (\u03BCm)')
    ax2.set_ylabel('Occurrence')
    prairie.style(ax2)

    ax3.plot(laser_position, 1e3*residuals, '.', color=color, markersize=1.5)
    ax3.set_ylim([-300, 300])
    ax3.set_title('Wire position error', loc='left')
    ax3.set_ylabel('Wire position error (\u03BCm)')
    ax3.set_xlabel('Laser position (mm)')
    prairie.style(ax3)

    plt.tight_layout()

    if separate is True:

        if save is True:
            plt.show(block=False)

            if saving_name is not None:
                plt.savefig(saving_name + '_calibration_residuals.png', format='png', dpi=DPI)
            else:
                print('saving_name is None - Figure not saved')
        else:
            plt.show(block=True)

        fig.clear()

    equation = "{:3.2f}".format(param[1]) + '-' + "{:3.2f}".format(param[2]) + '*' + 'cos(\u03C0-x+' + "{:3.2f}".format(
        param[0]) + ')'
    legend = 'Theoretical Wire position: ' + equation

    if separate is True:
        fig = plt.figure(figsize=(8, 2.5))
        ax1 = fig.add_subplot(111)
        prairie.use()

    ax1.plot(occlusion_position_mean, theorical_laser_position_mean, linewidth=0.5, color='black')
    ax1.plot(occlusion_position, laser_position, '.', color=color, markersize=4)
    ax1.legend([legend, 'Measured positions'])
    ax1.set_title('[' + folder_name.split('\\')[::-1][0] + '  ' + in_or_out + '] Theoretical wire positions vs. measured positions', loc='left')
    ax1.set_xlabel('Angular position at laser crossing (rad)')
    ax1.set_ylabel('Laser position (mm)')
    prairie.style(ax1)

    if complete_residuals_curve is True:

        ax4 = plt.subplot2grid((d, 2), (2, 0), colspan=2)
        ax4.plot(1e3*residuals, '.', color=color, markersize=1.5)
        ax4.plot(1e3*residuals, color=color, linewidth=0.5)
        ax4.set_title('Wire position error over scans', loc='left')
        ax4.set_ylabel('Wire position error (\u03BCm)')
        ax4.set_xlabel('Scan #')
        prairie.style(ax4)

    plt.tight_layout()

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            if separate is True:
                plt.savefig(saving_name + '_calibration_curve.png', format='png', dpi=DPI)
            else:
                plt.savefig(saving_name + '_calibration.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def plot_all_eccentricity(folder_name, in_or_out, howmuch='all', diagnostic=False, save=False, saving_name=None, separate=False):
    """
          Plot all the eccenticity curve extracted form the OPS processing to see if any jump occured

          Args:
              folder_name: folder containing PROCESSED_IN and PROCESSED_OUT, the processed data
              save: save the figure as png (figure will not be displayed if True)
              saving_name: Name of the png file (without extension)
          """

    if in_or_out is 'IN':
        filename = 'PROCESSED_IN.mat'
        color = '#018BCF'
    elif in_or_out is 'OUT':
        filename = 'PROCESSED_OUT.mat'
        color = '#0EA318'
    off = 100

    data = sio.loadmat(folder_name + '/' + filename, struct_as_record=False, squeeze_me=True)
    eccentricity = data['eccentricity']
    angular_position_SA = data['angular_position_SA']

    if diagnostic == True or diagnostic == 1:
        laser_position = data['laser_position']
        scan_number = data['scan_number']

    ref_ecc = eccentricity[0]
    ref_ecc = ref_ecc[off:ref_ecc.size - off]
    ref_pos = angular_position_SA[0]
    ref_pos = ref_pos[off:ref_pos.size - off]
    ecc_all = []

    def theor_ecc(x, a, b, c):
        return a * np.sin(x + b) + c

    popt, pcov = curve_fit(theor_ecc, ref_pos, ref_ecc, bounds=([-100, -100, -100], [100, 100, 100]))

    if howmuch == 'all':

        for ecc, pos in zip(eccentricity, angular_position_SA):
            ecc = ecc[off:ecc.size - off]
            pos = pos[off:pos.size - off]
            ecc = utils.resample(np.array([pos, ecc]), np.array([ref_pos, ref_ecc]))
            ecc_all.append(ecc[1])

        [ref_ecc, ref_pos] = [eccentricity[0], angular_position_SA[0]]

        deff = []
        residuals_mean = []

        if separate is True:
            fig = plt.figure(figsize=(8, 2.5))
            prairie.use()
            ax1 = fig.add_subplot(111)
        else:
            fig = plt.figure(figsize=(8, 4))
            prairie.use()
            ax2 = plt.subplot2grid((2, 2), (1, 1))
            ax3 = plt.subplot2grid((2, 2), (1, 0))
            ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

        ax1.plot(ref_pos, theor_ecc(ref_pos, popt[0], popt[1], popt[2]), linewidth=0.5, color='black')

        for ecc, pos in zip(eccentricity, angular_position_SA):
            ecc = ecc[off:ecc.size - off]
            pos = pos[off:pos.size - off]
            ax1.plot(pos, ecc, linewidth=0.8, color=color)

        if diagnostic == True or diagnostic == 1:

            for ecc, pos in zip(eccentricity, angular_position_SA):
                ecc = ecc[off:ecc.size - off]
                pos = pos[off:pos.size - off]
                # plt.plot(1e3*pos, ecc, linewidth=1)
                deft = np.abs(np.diff(ecc))
                deff.append(np.amax(deft))

        ax1.set_title('Position error and eccentricity compensation - Sensor A', loc='left')
        ax1.set_xlabel('Angular position (rad)')
        ax1.set_ylabel('Position error (rad)')
        ax1.legend(['Eccentricity global fit', 'Eccentricity profiles (' + str(eccentricity.size) + ')'])
        prairie.style(ax1)

        plt.tight_layout()

        if separate is True:

            if save is True:
                plt.show(block=False)

                if saving_name is not None:
                    plt.savefig(saving_name + '_curve.png', format='png', dpi=DPI)
                else:
                    print('saving_name is None - Figure not saved')
            else:
                plt.show(block=True)

            fig.clear()

            fig = plt.figure(figsize=(8, 2.5))
            prairie.use()
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax2 = fig.add_subplot(gs[0])
            ax3 = fig.add_subplot(gs[1])

        for ecc, pos in zip(eccentricity, angular_position_SA):
            ecc = ecc[off:ecc.size - off]
            pos = pos[off:pos.size - off]
            residuals = ecc - theor_ecc(pos, popt[0], popt[1], popt[2])
            ax2.plot(pos, 1e6*residuals, linewidth=0.2, color=color)
            residuals_mean.append(np.mean(residuals))

        ax2.set_title('Position error after compensation', loc='left')
        ax2.set_xlabel('Angular position (rad)')
        ax2.set_ylabel('Position error (\u03BCrad)')
        ax2.legend(['Residuals profiles (' + str(eccentricity.size) + ')'])
        prairie.style(ax2)

        make_histogram(1e6 * np.asarray(residuals_mean), [-10, 10], '\u03BCrad', ax3, projected=True, color=color)
        ax3.set_title('Error histogram (' + str(len(eccentricity)) + ' traces)', loc='left')
        ax3.set_ylabel('Occurrence')
        ax3.set_xlabel('Position error (\u03BCrad)')
        prairie.style(ax2)
        prairie.style(ax3)

        plt.tight_layout()

        if save is True:
            plt.show(block=False)

            if saving_name is not None:
                if separate is True:
                    plt.savefig(saving_name + '_residuals.png', format='png', dpi=DPI)
                else:
                    plt.savefig(saving_name + '.png', format='png', dpi=DPI)
            else:
                print('saving_name is None - Figure not saved')
        else:
            plt.show(block=True)

        # if diagnostic == True or diagnostic == 1:
        #     return [laser_position[np.argmax(np.asarray(deff))], scan_number[np.argmax(np.asarray(deff))]]

    elif howmuch == 'random':

        plt.figure()
        plt.subplot(1, 2, 1)
        prairie.use()()()
        prairie.use_colors('cold')
        plt.plot(ref_pos, theor_ecc(ref_pos, popt[0], popt[1], popt[2]), linewidth=0.5, color='black')
        plt.plot(ref_pos, ref_ecc, linewidth=1)
        plt.subplot(1, 2, 2)
        plt.plot(ref_pos, ref_ecc - theor_ecc(ref_pos, popt[0], popt[1], popt[2]), linewidth=0.5)
        plt.plot(ref_pos, utils.ref_ecc - theor_ecc(ref_pos, popt[0], popt[1], popt[2]))
        plt.plot(ref_pos, ref_ecc - theor_ecc(ref_pos, popt[0], popt[1], popt[2]))
        plt.show()

        return popt

    return ax1


def plot_all_speed(folder_name, in_or_out, save=False, saving_name=None, separate=True):

    if separate is True:
        fig = plt.figure(figsize=(8, 2.5))
        prairie.use()
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(8, 4))
        prairie.use()
        ax2 = plt.subplot2grid((2, 2), (1, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

    if in_or_out is 'IN':
        filename = 'PROCESSED_IN.mat'
        color = '#018BCF'
    elif in_or_out is 'OUT':
        filename = 'PROCESSED_OUT.mat'
        color = '#0EA318'

    data = sio.loadmat(folder_name + '/' + filename, struct_as_record=False, squeeze_me=True)
    speed_SA = data['speed_SA']
    time_SA = data['time_SA']

    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)

    ref_speed = speed_SA[0]
    ref_time = time_SA[0]
    R_speed = []
    M_speed = ref_speed

    offset = 100

    for speed, time in zip(speed_SA, time_SA):
        speed = speed[offset:(speed.size-offset)]
        time = time[offset:(time.size - offset - 1)]
        r_speed = utils.resample(np.array([time, speed]), np.array([ref_time, ref_speed]))
        M_speed = np.add(r_speed[1][0:M_speed.size], M_speed)
        M_speed /= 2

    ax1.plot(ref_time[0:M_speed.size], M_speed, 'k', linewidth=0.5)

    for speed, time in zip(speed_SA, time_SA):
        speed = speed[offset:(speed.size-offset)]
        time = time[offset:(time.size - offset - 1)]
        r_speed = utils.resample(np.array([time, speed]), np.array([ref_time, ref_speed]))
        ax1.plot(time, speed, linewidth=0.5, color=color)
        R_speed.append(np.mean((r_speed[1][0:M_speed.size] - M_speed)[200:M_speed.size-offset-20]))

    ax1.set_title('Speed profiles', loc='left')
    ax1.set_ylabel('Speed (rad/s)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(['Mean speed', 'Speed profiles (' + str(speed_SA.size) + ')'])
    prairie.style(ax1)

    plt.tight_layout()

    if separate is True:

        if save is True:
            plt.show(block=False)

            if saving_name is not None:
                plt.savefig(saving_name + '_curve.png', format='png', dpi=DPI)
            else:
                print('saving_name is None - Figure not saved')
        else:
            plt.show(block=True)

        fig.clear()

        fig = plt.figure(figsize=(8, 2.5))
        prairie.use()
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax2 = fig.add_subplot(gs[0])
        ax3 = fig.add_subplot(gs[1])

    for speed, time in zip(speed_SA, time_SA):
        speed = speed[offset:(speed.size-offset)]
        time = time[offset:(time.size - offset - 1)]
        r_speed = utils.resample(np.array([time, speed]), np.array([ref_time, ref_speed]))
        ax2.plot(r_speed[0][200:M_speed.size-offset-20], (r_speed[1][0:M_speed.size] - M_speed)[200:M_speed.size-offset-20], color=color, linewidth=0.5)
        R_speed.append(np.mean((r_speed[1][0:M_speed.size] - M_speed)[200:M_speed.size-offset-20]))

    make_histogram(np.asarray(R_speed), [-5, 5], 'rad/s', ax3, color=color)

    ax2.set_title('Speed profile residuals \n w.r.t the mean speed profile', loc='left')
    ax2.set_ylabel('Residuals (rad/s)')
    ax2.set_xlabel('Time (s)')
    ax2.legend(['Residuals profiles (' + str(speed_SA.size) + ')'])
    prairie.style(ax2)
    ax3.set_title('Error histogram', loc='left')
    ax3.set_ylabel('Occurrence')
    ax3.set_xlabel('Speed error (rad/s)')
    prairie.style(ax3)

    plt.tight_layout()

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            if separate is True:
                plt.savefig(saving_name + '_residuals.png', format='png', dpi=DPI)
            else:
                plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def plot_RDS(folder_name, in_or_out, offset=0, index=None):
    """
        Plot of the relative distance thresholding performed in the OPS processing to correct defaults

        Args:
            folder_name: folder containing PROCESSED_IN and PROCESSED_OUT, the processed data
            in_or_out: 'IN' or 'OUT' analysis
            offset : number of point to remove at the beginning and the end of the data
            index: index of the specific scan to analyse
    """

    if in_or_out is 'IN':
        filename = 'PROCESSED_IN.mat'
    elif in_or_out is 'OUT':
        filename = 'PROCESSED_OUT.mat'

    data = sio.loadmat(folder_name + '/' + filename, struct_as_record=False, squeeze_me=True)
    time_SA = data['time_SA']
    time_SB = data['time_SB']
    print(time_SA.size)

    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)
    rdcp = eval(config.get('OPS processing parameters', 'relative_distance_correction_prameters'))

    fig = plt.figure()
    prairie.use()()()
    ax = fig.add_subplot(111)
    plt.axhspan(rdcp[1], rdcp[0], color='black', alpha=0.1)

    if index == None:
        for i in np.arange(0, time_SA.size):
            distances_A = np.diff(time_SA[i])[offset:time_SA[i].size - 1 - offset]
            rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
            distances_B = np.diff(time_SB[i])[offset:time_SB[i].size - 1 - offset]
            rel_distances_B = np.divide(distances_B[1::], distances_B[0:distances_B.size - 1])
            ax.plot(1e3 * time_SA[i][offset:time_SA[i].size - 2 - offset], rel_distances_A, '.',
                     color=prairie.use_colors('cold', i))
            ax.plot(1e3 * time_SB[i][offset:time_SB[i].size - 2 - offset], rel_distances_B, '.',
                     color=prairie.use_colors('hot', i))
    else:
        i = index
        distances_A = np.diff(time_SA[i])[offset:time_SA[i].size - 1 - offset]
        rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
        distances_B = np.diff(time_SB[i])[offset:time_SB[i].size - 1 - offset]
        rel_distances_B = np.divide(distances_B[1::], distances_B[0:distances_B.size - 1])
        ax.plot(1e3 * time_SA[i][offset:time_SA[i].size - 2 - offset], rel_distances_A, '.',
                 color=prairie.use_colors('cold', i))
        ax.plot(1e3 * time_SB[i][offset:time_SB[i].size - 2 - offset], rel_distances_B, '.',
                 color=prairie.use_colors('hot', i))

    ax.set_xlabel('Time (' + '\u03BC' + 's)')
    ax.set_ylabel('Relative distance')
    ax.set_title('RDS plot', loc='left')
    red_patch = mpatches.Patch(color=prairie.use_colors('cold', i), label='Sensor A')
    blue_patch = mpatches.Patch(color=prairie.use_colors('hot', i), label='Sensor B')
    ax.legend(handles=[blue_patch, red_patch])
    prairie.style(ax)
    plt.show()


def plot_all_positions(folder_name, in_or_out, save=False, saving_name=None):

    fig = plt.figure(figsize=(8, 2.5))
    prairie.use()
    ax1 = fig.add_subplot(111)

    if in_or_out is 'IN':
        filename = 'PROCESSED_IN.mat'
        color = '#018BCF'
    elif in_or_out is 'OUT':
        filename = 'PROCESSED_OUT.mat'
        color = '#0EA318'

    data = sio.loadmat(folder_name + '/' + filename, struct_as_record=False, squeeze_me=True)
    angular_position_SA = data['angular_position_SA']
    time_SA = data['time_SA']

    for pos, time in zip(angular_position_SA, time_SA):
        ax1.plot(time[2:time.size - 2], pos[2:pos.size - 2], linewidth=0.5, color=color)

    ax1.set_title('Motor angular positions', loc='left')
    ax1.set_ylabel('Position (rad)')
    ax1.set_xlabel('Time (s)')
    ax1.legend(['position profiles (' + str(angular_position_SA.size) + ')'])
    prairie.style(ax1)

    plt.tight_layout()

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)
    

# Raw data analysis (TDMS)  ------------------------------------------------------------------------


def make_one_plot_analysis_from_tdms(INorOUT, datarange, data_SA, data_SB, data_PD, subplot, title, showplot):
    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)
    sampling_frequency = config.getfloat('OPS processing parameters', 'sampling_frequency')

    print(sampling_frequency)

    if INorOUT is 'IN':
        StartTime = datarange[0]
        EndTime = datarange[1]
        IndexStart = int(sampling_frequency * datarange[0])
        IndexStop = int(sampling_frequency * datarange[1])
        colors = ['#ece7f2', '#2b8cbe', '#2D3A40']
        IN = True

    elif INorOUT is 'OUT':
        StartTime = datarange[2]
        EndTime = datarange[3]
        IndexStart = int(sampling_frequency * datarange[2])
        IndexStop = int(sampling_frequency * datarange[3])
        colors = ['#ffeda0', '#f03b20', '#2D3A40']
        IN = False

    data_SA = data_SA[IndexStart:IndexStop]
    data_PD = data_PD[IndexStart:IndexStop]

    Data_SA = ops.process_position(data_SA, parameter_file, 0, showplot)
    Data_SB = ops.process_position(data_SB, parameter_file, 0, False)
    Data_SB_R = utils.resample(Data_SB, Data_SA)
    time_SA = Data_SA[0]
    time_SB = Data_SB[0]
    offset = 100

    eccentricity = np.subtract(Data_SA[1], Data_SB_R[1]) / 2
    #
    data_SA = utils.butter_lowpass_filter(data_SA, 1e6, 20e6, order=6)
    data_PD = utils.butter_lowpass_filter(data_PD, 1e6, 20e6, order=6)

    #    plt.subplot(subplot[0], subplot[1], subplot[2])
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(StartTime + np.arange(0, data_SA.size, 5) / sampling_frequency,
             data_SA[0::5] / np.amax(data_SA[0::5]) * np.amax(Data_SA[1]))
    plt.plot(StartTime + Data_SA[0], Data_SA[1])
    plt.plot(StartTime + np.arange(0, data_SA.size, 10) / sampling_frequency,
             data_PD[0::10] / np.amax(data_PD[0::10]) * np.amax(Data_SA[1]))
    plt.xlabel('Time')
    plt.ylabel('Angular Position')
    plt.xlim = [StartTime, EndTime]
    plt.title(title)

    plt.subplot(2, 2, 2)
    plt.plot(Data_SA[0] * 1e3, eccentricity)
    plt.title(title)

    plt.subplot(2, 2, 3)
    distances_A = np.diff(time_SA)[offset:time_SA.size - 1 - offset]
    rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
    plt.plot(1e3 * time_SA[offset:time_SA.size - 2 - offset], rel_distances_A, '.', color=prairie.use_colors('cold', 0))

    plt.show(block=False)

    _occ = StartTime + ops.find_occlusions(data_PD, IN, diagnostic_plot=False) / sampling_frequency
    occ = np.mean(_occ)
    p = utils.resample(Data_SA, np.array([[occ], [0]]))
    p = p[1]

    return _occ


def make_one_plot_analysis_from_tdms_2(INorOUT, datarange, data_SA, data_SB, data_PD, subplot, title, showplot, arange):

    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)
    range_55rs = eval(config.get('OPS processing parameters', 'range_55rs'))
    range_133rs = eval(config.get('OPS processing parameters', 'range_133rs'))
    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))

    if INorOUT is 'IN':
        print('------- OPS Processing IN -------')
        Index_for_StartTime = 0
        StartTime = 0
        EndTime = 0.08
        IN = True
    elif INorOUT is 'OUT':
        Index_for_StartTime = 2
        StartTime = 0.380
        EndTime = 0.460
        print('------- OPS Processing OUT -------')
        IN = False

    # ==========================================================================
    # Raw data file names extraction
    # ==========================================================================


    _data_SA = data_SA
    _data_SB = data_SB
    _data_PD = data_PD
    offset = 100

    Data_SA = ops.process_position(_data_SA, utils.resource_path('data/parameters.cfg'), StartTime, showplot)
    Data_SB = ops.process_position(_data_SB, utils.resource_path('data/parameters.cfg'), StartTime, showplot)

    Data_SB_R = utils.resample(Data_SB, Data_SA)

    # Eccentricity from OPS processing and saving in list
    _eccentricity = np.subtract(Data_SA[1], Data_SB_R[1]) / 2

    Data_SA[1] = np.subtract(Data_SA[1], _eccentricity)
    Data_SB_R[1] = np.add(Data_SB_R[1], _eccentricity)

    time_SA = Data_SA[0]
    time_SB = Data_SB[0]

    # Finding of occlusions and saving into a list
    _time_PD = StartTime + np.arange(0, _data_PD.size) * 1 / sampling_frequency
    occlusions = ops.find_occlusions(_data_PD, IN, True, StartTime)

    Data_SA_R = utils.resample(Data_SA, np.array([_time_PD, _data_PD]))
    occ1 = Data_SA_R[1][int(occlusions[0])]
    occ2 = Data_SA_R[1][int(occlusions[1])]
    _occlusion = (occ2 - occ1) / 2 + occ1

    print(_time_PD[int(np.mean(occlusions))])
    print(Data_SA_R[0][int(np.mean(occlusions))])
    print(Data_SA_R[1][int(np.mean(occlusions))])

    prairie.use()()()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(StartTime + np.arange(0, data_SA.size, 5) / sampling_frequency,
             data_SA[0::5] / np.amax(data_SA[0::5]) * np.amax(Data_SA[1]))
    plt.plot(Data_SA[0], Data_SA[1])
    plt.plot(StartTime + np.arange(0, data_SA.size, 10) / sampling_frequency,
             data_PD[0::10] / np.amax(data_PD[0::10]) * np.amax(Data_SA[1]))
    plt.xlabel('Time')
    plt.ylabel('Angular Position')
    plt.xlim = [StartTime, EndTime]
    plt.title(title)

    plt.subplot(2, 2, 2)
    plt.plot(Data_SA[0] * 1e3, _eccentricity)
    plt.title(title)

    plt.subplot(2, 2, 3)
    distances_A = np.diff(time_SA)[offset:time_SA.size - 1 - offset]
    rel_distances_A = np.divide(distances_A[1::], distances_A[0:distances_A.size - 1])
    plt.plot(1e3 * time_SA[offset:time_SA.size - 2 - offset], rel_distances_A, '.', color=prairie.use_colors('cold', 0))

    plt.show(block=False)

    return _occlusion


def plot_analysis_from_tdms(file, showplot=False):
    """
      Analysis of a single tdms file for the data acquired during the second calibration campaign (After the
      LabView program modification concerning the time windows of INs and OUTs)

      Args:
          file: tdms file to analyse
          showplot: show OPS processing plots

      Returns
          Values of crossing position
      """
    
    file = utils.reformate_path(file)

    tdms_file = TdmsFile(file)
    data__s_a = tdms_file.object('Picoscope Data', 'DISC PHOTODIODE HOME').data
    data__s_b = tdms_file.object('Picoscope Data', 'DISC PHOTODIODE IN').data
    data__p_d = np.abs(-tdms_file.object('Picoscope Data', 'WIRE PHOTODIODE').data)

    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)
    range_55rs = eval(config.get('OPS processing parameters', 'range_55rs'))
    range_133rs = eval(config.get('OPS processing parameters', 'range_133rs'))

    if file.find('55rs') != -1:
        datarange = range_55rs
    elif file.find('133rs') != -1:
        datarange = range_133rs

    a1 = make_one_plot_analysis_from_tdms('IN', datarange, data__s_a, data__s_b, data__p_d, [2, 2, 1], 'OPSA_IN', showplot)
    a2 = make_one_plot_analysis_from_tdms('OUT', datarange, data__s_a, data__s_b, data__p_d, [2, 2, 2], 'OPSA_OUT', showplot)
    a3 = make_one_plot_analysis_from_tdms('IN', datarange, data__s_b, data__s_b, data__p_d, [2, 2, 3], 'OPSB_IN', showplot)
    a4 = make_one_plot_analysis_from_tdms('OUT', datarange, data__s_b, data__s_b, data__p_d, [2, 2, 4], 'OPSB_OUT', showplot)

    return [a1, a2, a3, a4]


def plot_analysis_from_tdms_2(file, show_plot=False):
    """
    Analysis of a single tdms file for the data acquired during the second calibration campaign (After the
    LabView program modification concerning the time windows of INs and OUTs)

    Args:
        file: tdms file to analyse
        show_plot: show OPS processing plots

    Returns
        Values of crossing position
    """

    file = utils.reformate_path(file)

    parameter_file = utils.resource_path('data/parameters.cfg')
    config = configparser.RawConfigParser()
    config.read(parameter_file)
    range_55rs = eval(config.get('OPS processing parameters', 'range_55rs'))
    range_133rs = eval(config.get('OPS processing parameters', 'range_133rs'))
    sampling_frequency = eval(config.get('OPS processing parameters', 'sampling_frequency'))
    start_time_out = int(sampling_frequency * 0.020)
    print(start_time_out)

    tdms_file = TdmsFile(file)
    data__s_a__in = tdms_file.object('Picoscope Data', 'DISC PH. HOME dir IN').data
    data__s_b__in = tdms_file.object('Picoscope Data', 'DISC PH. IN dir IN').data
    data__p_d__in = np.abs(-tdms_file.object('Picoscope Data', 'WIRE PH. dir IN').data)
    data__s_a__out = tdms_file.object('Picoscope Data', 'DISC PH. HOME dir HOME').data[start_time_out::]
    data__s_b__out = tdms_file.object('Picoscope Data', 'DISC PH. IN dir HOME').data[start_time_out::]
    data__p_d__out = np.abs(-tdms_file.object('Picoscope Data', 'WIRE PH. dir HOME').data)[start_time_out::]

    if file.find('55rs') != -1:
        data_range = range_55rs
    elif file.find('133rs') != -1:
        data_range = range_133rs

    data_range = np.asarray(data_range)

    a1 = make_one_plot_analysis_from_tdms_2('IN', data_range, data__s_a__in, data__s_b__in, data__p_d__in, [2, 2, 1], 'OPSA_IN',
                                            show_plot, data_range[0:2] / sampling_frequency)
    a2 = make_one_plot_analysis_from_tdms_2('OUT', data_range, data__s_a__out, data__s_b__out, data__p_d__out, [2, 2, 2],
                                            'OPSA_OUT', show_plot, data_range[2:4] / sampling_frequency)
    a3 = make_one_plot_analysis_from_tdms_2('IN', data_range, data__s_b__in, data__s_b__in, data__p_d__in, [2, 2, 3], 'OPSB_IN',
                                            show_plot, data_range[0:2] / sampling_frequency)
    a4 = make_one_plot_analysis_from_tdms_2('OUT', data_range, data__s_b__out, data__s_b__out, data__p_d__out, [2, 2, 4],
                                            'OPSB_OUT', show_plot, data_range[2:4] / sampling_frequency)

    return [a1, a2, a3, a4]


def plot_fatigue_monitoring(folders):
    blue = '#3C5D9D'
    orange = '#FA7D30'

    labels = []
    dates = []
    a = []
    b = []
    c = []
    colors = []
    sigma_IN = []
    center_IN = []
    sigma_OUT = []
    center_OUT = []
    size = (8, 2.5)

    day_max=16

    for folder in folders:

        date = float(folder.split('/')[::-1][0].split('__')[1].split('_')[2])
        plot_legend = folder.split('/')[::-1][0].split('__')[1].split('_')[1] + '/' + \
                      folder.split('/')[::-1][0].split('__')[1].split('_')[2]

        color = int(folder.split('/')[::-1][0].split('__')[1].split('_')[2])

        if int(folder.split('/')[::-1][0].split('__')[2].split('_')[0]) <= 12:
            plot_legend += ' a.m.'
        else:
            plot_legend += ' p.m.'
            date += 0.5
            color += 0.5

        labels.append(plot_legend)
        data = sio.loadmat(folder + '/' + 'calibration_results.mat', struct_as_record=False, squeeze_me=True)
        p = data['f_parameters_IN']
        dates.append(2 * date)
        a.append(p[0])
        b.append(p[1])
        c.append(p[2])
        acolor = np.zeros(3)
        acolor[1] = (day_max - color) / day_max
        acolor[2] = 0.75
        acolor[0] = 0.2
        colors.append(acolor)
        sigma_IN.append(data['sigma_IN'])
        center_IN.append(data['center_IN'])
        sigma_OUT.append(data['sigma_OUT'])
        center_OUT.append(data['center_OUT'])

    prairie.use()

    center_IN = np.asarray(center_IN)
    center_OUT = np.asarray(center_OUT)

    fig, ax1 = plt.subplots(figsize=size)
    ax1.plot(dates, sigma_IN, '.', color=blue)
    ax1.set_xlabel('Date')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('um', color=blue)
    ax1.tick_params('y', colors=blue)
    prairie.style(ax1)
    plt.xticks(dates, labels, rotation=30)
    plt.ylim([6, 15])
    plt.title('Fatigue test monitoring of June 2017 (IN) - Angular position error sigma')

    # ax2 = ax1.twinx()
    # s2 = np.array([600, 4900, 5200, 10000, 10300, 14600, 14900, 20400, 20700, 25500]) + 8500
    # ax2.plot(t, s2, color=orange, alpha=0.3)
    # ax2.set_ylabel('Number of scans', color=orange)
    # ax2.tick_params('y', colors=orange)

    fig.tight_layout()
    plt.show(block=False)

    plt.savefig('../figures/Fatigue test monitoring of June 2017 - Angular position error sigma',
                format='png', dpi=DPI)

    fig, ax1 = plt.subplots(figsize=size)
    ax1.plot(dates, center_IN - center_IN[0], '.', color=blue)
    ax1.set_xlabel('Date')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('um', color=blue)
    ax1.tick_params('y', colors=blue)
    prairie.style(ax1)
    plt.xticks(dates, labels, rotation=30)
    plt.ylim([-10, 30])
    plt.title('Fatigue test monitoring of June 2017 (IN) - Relative position of the center')

    # ax2 = ax1.twinx()
    # s2 = np.array([600, 4900, 5200, 10000, 10300, 14600, 14900, 20400, 20700, 25500]) + 8500
    # ax2.plot(t, s2, color=orange, alpha=0.3)
    # ax2.set_ylabel('Number of scans', color=orange)
    # ax2.tick_params('y', colors=orange)

    fig.tight_layout()
    plt.show(block=False)

    plt.savefig('../figures/Fatigue test monitoring of June 2017 (IN) - Relative position of the center',
                format='png', dpi=DPI)

    fig, ax1 = plt.subplots(figsize=size)
    ax1.plot(dates, sigma_OUT, '.', color=blue)
    ax1.set_xlabel('Date')
    # Make the y-axis label, ticks and tick labels match the lOUTe color.
    ax1.set_ylabel('um', color=blue)
    ax1.tick_params('y', colors=blue)
    prairie.style(ax1)
    plt.xticks(dates, labels, rotation=30)
    plt.ylim([6, 15])
    plt.title('Fatigue test monitoring of June 2017 (OUT) - Angular position error sigma')

    # ax2 = ax1.twOUTx()
    # s2 = np.array([600, 4900, 5200, 10000, 10300, 14600, 14900, 20400, 20700, 25500]) + 8500
    # ax2.plot(t, s2, color=orange, alpha=0.3)
    # ax2.set_ylabel('Number of scans', color=orange)
    # ax2.tick_params('y', colors=orange)

    fig.tight_layout()
    plt.show(block=False)

    plt.savefig('../figures/Fatigue test monitoring of June 2017 (OUT) - Angular position error sigma.png',
                format='png', dpi=DPI)

    fig, ax1 = plt.subplots(figsize=size)
    ax1.plot(dates, center_OUT, '.', color=blue)
    ax1.set_xlabel('Date')
    # Make the y-axis label, ticks and tick labels match the lOUTe color.
    ax1.set_ylabel('um', color=blue)
    ax1.tick_params('y', colors=blue)
    prairie.style(ax1)
    plt.xticks(dates, labels, rotation=30)
    plt.ylim([-10, 30])
    plt.title('Fatigue test monitorOUTg of June 2017 (OUT) - Relative position of the center')

    # ax2 = ax1.twOUTx()
    # s2 = np.array([600, 4900, 5200, 10000, 10300, 14600, 14900, 20400, 20700, 25500]) + 8500
    # ax2.plot(t, s2, color=orange, alpha=0.3)
    # ax2.set_ylabel('Number of scans', color=orange)
    # ax2.tick_params('y', colors=orange)

    fig.tight_layout()
    plt.show(block=False)

    plt.savefig('../figures/Fatigue test monitorOUTg of June 2017 (OUT) - Relative position of the center.png',
                format='png', dpi=DPI)


# Multiple calibration analysis  ---------------------------------------------------------------------


def plot_superposed_theoretical_curve(folders, in_or_out, save=False, saving_name=None):
    """
    Plot the superposition of several calibration fitted curves

    Args:
        folders: list of PROCESSED data folders
        save: save the figure as png (figure will not be displayed if True)

    """
    day_max = 20

    fig = plt.figure(figsize=(9, 5))
    prairie.use()

    f = []
    legend = []
    residuals = []
    rms_list = []
    plot_colors = []

    ops.mean_fit_parameters(folders)

    portion = np.arange(0.65, 1.65, 0.01)
    mean_fit_file = sio.loadmat('../data/mean_fit.mat', struct_as_record=False, squeeze_me=True)

    if in_or_out is 'IN':
        a = mean_fit_file['a_IN']
        b = mean_fit_file['b_IN']
        c = mean_fit_file['c_IN']
    elif in_or_out is 'OUT':
        a = mean_fit_file['a_OUT']
        b = mean_fit_file['b_OUT']
        c = mean_fit_file['c_OUT']

    f0 = utils.theoretical_laser_position(portion, a, b, c)

    ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)

    for folder in folders:

        data = sio.loadmat(folder + '/' + 'calibration_results.mat', struct_as_record=False, squeeze_me=True)

        if in_or_out is 'IN':
            p = data['f_parameters_IN']
        elif in_or_out is 'OUT':
            p = data['f_parameters_OUT']

        plot_legend = folder.split('/')[::-1][0].split('__')[1].split('_')[1] + '/' + \
                      folder.split('/')[::-1][0].split('__')[1].split('_')[2]

        color = int(folder.split('/')[::-1][0].split('__')[1].split('_')[2])

        if int(folder.split('/')[::-1][0].split('__')[2].split('_')[0]) <= 12:
            plot_legend += ' a.m.'
        else:
            plot_legend += ' p.m.'
            color += 0.5

        theoretical_laser_position = utils.theoretical_laser_position(portion, p[0], p[1], p[2])

        RMS = 1e3 * sqrt(mean_squared_error(theoretical_laser_position, f0))
        residuals.append(1e3 * (theoretical_laser_position - f0))
        legend.append(plot_legend + ': ' + "{:.1f}".format(RMS))
        rms_list.append(RMS)

        plot_color = np.zeros(3)
        plot_color[1] = (day_max - color) / day_max
        plot_color[2] = 0.75
        plot_color[0] = 0.2
        plot_colors.append(plot_color)
        ax.plot(portion, theoretical_laser_position, color=plot_color)
        # plt.text(1 + u * 1e-9, utils.theoretical_laser_position(1, p[0], p[1], p[2]) + 0.0001, plot_legend, fontsize=8, color=acolor)

    _legend = plt.legend(legend, title='Relative difference' + '(\u03BCm)')
    plt.setp(_legend.get_title(), fontsize=8)
    ax.set_xlabel('Angular wire position (rad)')
    ax.set_ylabel('Laser Position (mm)')
    ax.set_title('Mechanical equation of the scanner over calibrations' + ' ' + in_or_out, fontsize=9, weight='bold', family='Arial',
              loc='left')
    prairie.style(ax)

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    for d, color in zip(residuals, plot_colors):
        ax2.plot(portion, d, color=color)
    ax2.set_ylim([-150, 150])
    ax2.set_ylabel('RMS error (\u03BCm)')
    ax2.set_xlabel('Angular wire position (rad)')
    plt.title('Mechanical equation residuals (w.r.t the mean curve)', loc='left')
    prairie.style(ax2)

    ax3 = plt.subplot2grid((2, 2), (1, 1))
    make_histogram(rms_list, [-100, 100], '\u03BCm')
    prairie.style(ax3)
    ax3.set_xlabel('RMS error (\u03BCm)')
    ax3.set_ylabel('Normalized occurence')
    plt.title('histogram of rms error', loc='left')

    plt.tight_layout()

    if save is True:
        plt.show(block=False)
        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def plot_multiple_calibrations_analysis(folders, in_or_out='IN', save=False, saving_name=None):

    fig = plt.figure(figsize=(10, 4))
    prairie.use()
    day_max = 25

    legends = []

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    labels = []
    dates = []
    colors = []
    mu = []
    ranges_0 = []
    ranges_1 = []

    for folder in folders:

        if os.path.exists(folder + '/calibration_results.mat'):

            date = float(folder.split('/')[::-1][0].split('__')[1].split('_')[2])
            plot_legend = folder.split('/')[::-1][0].split('__')[1].split('_')[1] + '/' + \
                          folder.split('/')[::-1][0].split('__')[1].split('_')[2]

            color = int(folder.split('/')[::-1][0].split('__')[1].split('_')[2])

            if int(folder.split('/')[::-1][0].split('__')[2].split('_')[0]) <= 12:
                plot_legend += ' a.m.'
            else:
                plot_legend += ' p.m.'
                date += 0.5
                color += 0.5

            labels.append(plot_legend)
            data = sio.loadmat(folder + '/calibration_results.mat', struct_as_record=False, squeeze_me=True)
            dates.append(2 * date)
            acolor = np.zeros(3)
            acolor[1] = (day_max - color) / day_max
            acolor[2] = 0.75
            acolor[0] = 0.2
            colors.append(acolor)

            if in_or_out is 'IN':

                residuals = np.asarray(data['residuals_IN_origin'])
                range_0 = 1e3 * (np.mean(residuals) - 4 * np.std(residuals))
                ranges_0.append(range_0)
                range_1 = 1e3 * (np.mean(residuals) + 4 * np.std(residuals))
                ranges_1.append(range_1)

                ax1.plot(data['laser_position_IN'], 1e3 * data['residuals_IN_origin'], '.', color=acolor)
                m = make_histogram(1e3 * data['residuals_IN_origin'], [range_0, range_1], 'um', axe=ax2,
                                      color=acolor)

            elif in_or_out is 'OUT':

                residuals = np.asarray(data['residuals_OUT_origin'])
                range_0 = 1e3 * (np.mean(residuals) - 4 * np.std(residuals))
                ranges_0.append(range_0)
                range_1 = 1e3 * (np.mean(residuals) + 4 * np.std(residuals))
                ranges_1.append(range_1)

                ax1.plot(data['laser_position_OUT'], 1e3 * data['residuals_OUT_origin'], '.', color=acolor)
                m = make_histogram(1e3 * data['residuals_OUT_origin'], [range_0, range_1], 'um', axe=ax2,
                                      color=acolor)
            mu.append(m)

    if len(ranges_0) > 0:
        ax1.set_ylim([np.min(np.asarray(ranges_0)), np.max(np.asarray(ranges_1))])
        ax2.set_xlim([np.min(np.asarray(ranges_0)), np.max(np.asarray(ranges_1))])
    else:
        ax1.set_ylim([-100, 100])

    ax1.set_xlim([-60, 60])

    if len(folders) > 1:

        texts = ax2.get_legend().get_texts()
        for text in texts:
            legends.append(text._text)

        legends = np.asarray([label + '  ' + legend for label, legend in zip(labels, legends)])

        ax2.legend(legends, bbox_to_anchor=(1.05, 2.5), loc=2, borderpad=1)

        prairie.style(ax1)
        prairie.style(ax2)

        if in_or_out is 'IN':
            title = 'Wire position error overs scans - IN'
        else:
            title = 'Wire position error overs scans - OUT'

        ax1.set_title(title, loc='left')
        ax1.set_ylabel('Error (\u03BCm)')
        ax1.set_xlabel('Laser position (mm)')
        ax2.set_title('Wire position error histogram', loc='left')
        ax2.set_ylabel('Occurence')
        ax2.set_xlabel('Error (\u03BCm)')

        prairie.style(ax1)
        prairie.style(ax2)

        fig.tight_layout()
        fig.subplots_adjust(right=0.7)

        if save is True:
            plt.show(block=False)

            if saving_name is not None:
                plt.savefig(saving_name + '.png', format='png', dpi=DPI)
            else:
                print('saving_name is None - Figure not saved')
        else:
            plt.show(block=True)


def plot_global_residuals(folders, save=False, saving_name=None):

    legends = []

    fig = plt.figure(figsize=(8, 3))
    prairie.use()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    # ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((1, 2), (0, 1))
    # ax4 = plt.subplot2grid((2, 2), (1, 1))
    labels = []
    dates = []
    colors = []
    residuals_IN = np.empty((1))
    residuals_OUT = np.empty((1))
    laser_position_IN =  np.empty((1))
    laser_position_OUT = np.empty((1))
    mu = []
    sigma =[]

    day_max= len(folders)


    for folder in folders:

        color = int(folder.split('/')[::-1][0].split('__')[1].split('_')[2])

        if int(folder.split('/')[::-1][0].split('__')[2].split('_')[0]) >= 12:
            color += 0.5

        # acolor = np.zeros(3)
        # acolor[1] = (day_max - color) / day_max
        # acolor[2] = 0.75
        # acolor[0] = 0.2

        data = sio.loadmat(folder + '/' + 'calibration_results.mat', struct_as_record=False, squeeze_me=True)
        residuals_IN = np.concatenate((residuals_IN, data['residuals_IN_origin']), axis=0)
        residuals_OUT = np.concatenate((residuals_OUT, data['residuals_OUT_origin']), axis=0)
        laser_position_IN = np.concatenate((laser_position_IN, data['laser_position_IN']), axis=0)
        laser_position_OUT = np.concatenate((laser_position_OUT, data['laser_position_OUT']), axis=0)

        ax1.plot(data['laser_position_IN'], 1e3 * data['residuals_IN_origin'], '.', markersize=3.5)
        ax3.plot(data['laser_position_OUT'], 1e3 * data['residuals_OUT_origin'], '.', markersize=3.5)

    laser_position_IN_mean = []
    residuals_IN_mean = []

    laser_position_OUT_mean = []
    residuals_OUT_mean = []

    for laser_position in laser_position_IN:
        residuals_IN_mean.append(np.mean(residuals_IN[np.where(laser_position_IN == laser_position)]))
        laser_position_IN_mean.append(laser_position)

    for laser_position in laser_position_OUT:
        residuals_OUT_mean.append(np.mean(residuals_OUT[np.where(laser_position_OUT == laser_position)]))
        laser_position_OUT_mean.append(laser_position)


    laser_position_IN_mean = np.asarray(laser_position_IN_mean)
    residuals_IN_mean = np.asarray(residuals_IN_mean)

    laser_position_OUT_mean = np.asarray(laser_position_OUT_mean)
    residuals_OUT_mean = np.asarray(residuals_OUT_mean)

    # in_legend = make_histogram(1e3 * residuals_IN, [-200, 200], 'um')
    # out_legend = make_histogram(1e3 * residuals_OUT, [-200, 200], 'um')
    ax1.set_ylim([-200, 200])
    ax3.set_ylim([-200, 200])

    prairie.style(ax1)
    # prairie.style(ax2)
    prairie.style(ax3)
    # prairie.style(ax4)
    plt.tight_layout()
    ax1.set_title('Wire position error overs scans - IN', loc='left')
    ax1.set_ylabel('Error (\u03BCm)')
    ax1.set_xlabel('Laser position (mm)')
    ax1.legend(['\u03C3 ' + "{:3.3f}".format(np.std(1e3 * residuals_IN) / np.sqrt(2)) + '   ' + '\u03BC ' + "{:3.3f}".format(np.mean(1e3 * residuals_IN)) + '  (\u03BCm)'])
    ax3.set_title('Wire position error overs scans - OUT', loc='left')
    ax3.set_ylabel('Error (\u03BCm)')
    ax3.set_xlabel('Laser position (mm)')
    ax3.legend(['\u03C3 ' + "{:3.3f}".format(
        np.std(1e3 * residuals_OUT) / np.sqrt(2)) + '   ' + '\u03BC ' + "{:3.3f}".format(
        np.mean(1e3 * residuals_OUT)) + '  (\u03BCm)'])
    # ax2.set_title('Wire position error histogram - IN', loc='left')
    # ax2.set_ylabel('Occurence')
    # ax2.set_xlabel('Error (\u03BCm)')
    # ax4.set_title('Wire position error histogram - OUT', loc='left')
    # ax4.set_ylabel('Occurence')
    # ax4.set_xlabel('Error (\u03BCm)')
    plt.tight_layout()

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)


def plot_residuals_shape(folders, save=False, saving_name=None, origin=True, title=None):

    fig = plt.figure(figsize=(8, 3))
    prairie.use()
    ax = fig.add_subplot(111)

    M = []

    i = 0

    for folder in folders:
        data = sio.loadmat(folder + '/calibration_results.mat')
        residuals = data['residuals_IN_origin_mean'][0]
        laser_position = data['laser_position_IN_mean'][0]
        if i == 0:
            plt.plot(laser_position, utils.butter_lowpass_filter(residuals, 1 / 101, 1 / 10) - np.mean(residuals),
                     alpha=0.2, linewidth=2, label='Calibration residuals profiles (' + str(len(folders)) + ' - filtered and centered)')
        else:
            plt.plot(laser_position, utils.butter_lowpass_filter(residuals, 1 / 101, 1 / 10) - np.mean(residuals),
                     alpha=0.2, linewidth=2, label='_nolegend_')
        # plt.plot(laser_position, residuals - np.mean(residuals),
        #          alpha=0.2, linewidth=2, label='_nolegend_')
        M.append(residuals)
        i += 1

    M = np.asarray(M)
    M = np.mean(M, 0)

    ax.plot(laser_position, utils.butter_lowpass_filter(M, 1 / 101, 1 / 10), color='k', linewidth=2.5,
            label='Mean residual profile')
    # ax.plot(laser_position, M, color='k', linewidth=2.5,
    #         label='Mean residual profile')
    ax.set_xlabel('Laser position (mm)')
    ax.set_ylabel('Residual error (\u03BCm)')
    if title is not None:
        ax.set_title('Residuals shape of the ' + title, loc='left')
    ax.legend()
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


def plot_calibration_procedure(folders, save=False, saving_name=None, origin=True, title=None, strategy=1):

    fig = plt.figure(figsize=(8, 3))
    prairie.use()

    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    M = []

    R = []

    final_residuals = []

    i=0

    for folder in folders:
        data = sio.loadmat(folder + '/calibration_results.mat')
        residuals = 1e3*data['residuals_IN_origin'][0]
        laser_position = data['laser_position_IN'][0]
        if i==0:
            ax1.plot(laser_position, residuals, '.', alpha=0.3, label='Residuals with the theoretical curve')
        else:
            ax1.plot(laser_position, residuals, '.', alpha=0.3, label='_nolegend_')

        M.append(1e3*data['residuals_IN_origin_mean'][0])
        i += 1

    M = np.asarray(M)
    M = np.mean(M, 0)

    if strategy == 1:
        M = np.ones(M.size)*np.mean(M)

    for folder in folders:
        data = sio.loadmat(folder + '/calibration_results.mat')
        residuals = 1e3*data['residuals_IN_origin'][0]
        laser_position = data['laser_position_IN'][0]
        laser_position_mean = data['laser_position_IN_mean'][0]
        for l in laser_position_mean:
            R.append(np.asarray(residuals[np.where(laser_position == l)] - M[np.where(l == laser_position_mean)]))

    ax1.plot(data['laser_position_IN_mean'][0], M, alpha=0.5, linewidth=5, label='Raw calibration curve')
    ax1.plot(data['laser_position_IN_mean'][0], utils.butter_lowpass_filter(M, 1 / 101, 1 / 10), color='k', linewidth=1.5, label='Final Calibration curve')
    ax1.set_xlim([-50, 50])
    ax1.set_ylim([-80, 80])
    ax1.set_title('Wire position error with regard to the theoretical curve vs. the calibration curve')

    ax1.set_xlabel('Laser position (mm)')
    ax1.set_ylabel('Position error (\u03BCm)')
    ax1.legend(loc='upper right')
    plt.tight_layout()
    prairie.style(ax1)

    for folder in folders:
        data = sio.loadmat(folder + '/calibration_results.mat')
        residuals = data['residuals_IN_origin_mean'][0]
        final_residuals.append(utils.butter_lowpass_filter(M, 1 / 101, 1 / 10)-residuals)

    final_residuals = (np.hstack(R)).flatten()

    make_histogram(final_residuals, [-120, 120], '\u03BCm', axe=ax2)

    ax2.set_title('Wire position error histogram')
    ax2.set_xlabel('Position error (\u03BCm)')
    ax2.set_ylabel('Normalized occurrence')
    prairie.style(ax2)

    if save is True:
        plt.show(block=False)

        if saving_name is not None:
            plt.savefig(saving_name + '.png', format='png', dpi=DPI)
        else:
            print('saving_name is None - Figure not saved')
    else:
        plt.show(block=True)
        
        
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













