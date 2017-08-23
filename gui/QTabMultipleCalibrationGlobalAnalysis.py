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


from __future__ import unicode_literals

import os
import numpy as np
import scipy.io as sio

from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from lib import prairie
from gui.mplCanvas import mplCanvas
from lib import diagnostic_tools as dt


class QTabMultipleCalibrationGlobalAnalysis(QWidget):

    def __init__(self, parent=None):

        super(QTabMultipleCalibrationGlobalAnalysis, self).__init__(parent=None)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QWidget(self)

        self.in_or_out = 'IN'
        self.folders = []

        main_layout = QVBoxLayout(self.main_widget)
        self.plot = plot(self.main_widget, width=6.5, height=6, dpi=100)
        self.navi_toolbar = NavigationToolbar(self.plot, self)
        main_layout.addWidget(self.navi_toolbar)
        main_layout.addWidget(self.plot)

        self.setLayout(main_layout)

    def set_folder(self, folders, in_or_out):
        self.folders = folders
        self.in_or_out = in_or_out
        self.plot.folders = folders
        self.plot.in_or_out = in_or_out
        self.plot.compute_initial_figure()
        self.plot.draw()


class plot(mplCanvas):
    """Simple canvas with a sine plot."""

    def __init__(self, parent, width, height, dpi):

        self.folders = []

        self.ax1 = 0

        self.in_or_out = 'IN'

        super(plot, self).__init__(parent, width, height, dpi)

    def compute_initial_figure(self):

        self.fig.clear()

        prairie.use()
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax3 = self.fig.add_subplot(2, 1, 2)
        residuals_IN = np.empty(1)
        residuals_OUT = np.empty(1)
        laser_position_IN = np.empty(1)
        laser_position_OUT = np.empty(1)

        for folder in self.folders:

            if os.path.exists(folder + '/calibration_results.mat'):

                color = int(folder.split('/')[::-1][0].split('__')[1].split('_')[2])

                if int(folder.split('/')[::-1][0].split('__')[2].split('_')[0]) >= 12:
                    color += 0.5

                data = sio.loadmat(folder + '/' + 'calibration_results.mat', struct_as_record=False, squeeze_me=True)
                residuals_IN = np.concatenate((residuals_IN, data['residuals_IN_origin']), axis=0)
                residuals_OUT = np.concatenate((residuals_OUT, data['residuals_OUT_origin']), axis=0)
                laser_position_IN = np.concatenate((laser_position_IN, data['laser_position_IN']), axis=0)
                laser_position_OUT = np.concatenate((laser_position_OUT, data['laser_position_OUT']), axis=0)

                ax1.plot(data['laser_position_IN'], 1e3 * data['residuals_IN_origin'], '.')
                ax3.plot(data['laser_position_OUT'], 1e3 * data['residuals_OUT_origin'], '.')

        if len(self.folders) > 1:

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

            ax1.set_ylim([-100, 100])
            ax3.set_ylim([-100, 100])

            prairie.style(ax1)
            prairie.style(ax3)
            ax1.set_title('Wire position error overs scans - IN', loc='left')
            ax1.set_ylabel('Error (\u03BCm)')
            ax1.set_xlabel('Laser position (mm)')
            ax1.legend(['\u03C3 ' + "{:3.3f}".format(
                np.std(1e3 * residuals_IN) / np.sqrt(2)) + '   ' + '\u03BC ' + "{:3.3f}".format(
                np.mean(1e3 * residuals_IN)) + '  (\u03BCm)'])
            ax3.set_title('Wire position error overs scans - OUT', loc='left')
            ax3.set_ylabel('Error (\u03BCm)')
            ax3.set_xlabel('Laser position (mm)')
            ax3.legend(['\u03C3 ' + "{:3.3f}".format(
                np.std(1e3 * residuals_OUT) / np.sqrt(2)) + '   ' + '\u03BC ' + "{:3.3f}".format(
                np.mean(1e3 * residuals_OUT)) + '  (\u03BCm)'])
            self.fig.tight_layout()