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
from lib import utils
from gui.mplCanvas import mplCanvas


class QTabMultipleResidualsShape(QWidget):

    def __init__(self, parent=None):

        super(QTabMultipleResidualsShape, self).__init__(parent=None)

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
        residuals_IN = []
        residuals_OUT = []
        laser_position_IN = []
        laser_position_OUT = []

        for folder in self.folders:

            if os.path.exists(folder + '/calibration_results.mat'):

                color = int(folder.split('/')[::-1][0].split('__')[1].split('_')[2])

                if int(folder.split('/')[::-1][0].split('__')[2].split('_')[0]) >= 12:
                    color += 0.5

                data = sio.loadmat(folder + '/' + 'calibration_results.mat', struct_as_record=False, squeeze_me=True)
                residuals_IN.append(data['residuals_IN_origin_mean'])
                residuals_OUT.append(data['residuals_OUT_origin_mean'])
                laser_position_IN.append(data['laser_position_IN_mean'])
                laser_position_OUT.append(data['laser_position_OUT_mean'])

        if len(self.folders) > 1:

            M = []

            for residuals, laser_position in zip(residuals_IN, laser_position_IN):
                ax1.plot(laser_position, utils.butter_lowpass_filter(residuals, 1 / 101, 1 / 10) - np.mean(residuals),
                         alpha=0.2, linewidth=2, label='_nolegend_')
                M.append(residuals)

            M = np.asarray(M)
            M = np.mean(M, 0)

            ax1.plot(laser_position, utils.butter_lowpass_filter(M, 1 / 101, 1 / 10), color='k', linewidth=2.5,
                    label='Mean residual profile')
            ax1.set_xlabel('Laser position (mm)')
            ax1.set_ylabel('Residual error (\u03BCm)')
            ax1.legend()
            prairie.style(ax1)

            M = []

            for residuals, laser_position in zip(residuals_OUT, laser_position_OUT):
                ax3.plot(laser_position, utils.butter_lowpass_filter(residuals, 1 / 101, 1 / 10) - np.mean(residuals),
                         alpha=0.2, linewidth=2, label='_nolegend_')
                M.append(residuals)

            M = np.asarray(M)
            M = np.mean(M, 0)

            ax3.plot(laser_position, utils.butter_lowpass_filter(M, 1 / 101, 1 / 10), color='k', linewidth=2.5,
                    label='Mean residual profile')
            ax3.set_xlabel('Laser position (mm)')
            ax3.set_ylabel('Residual error (\u03BCm)')
            ax3.legend()
            prairie.style(ax3)