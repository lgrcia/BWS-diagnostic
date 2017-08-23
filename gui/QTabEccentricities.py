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

import numpy as np

from PyQt5 import QtCore
from scipy.optimize import curve_fit
from PyQt5.QtWidgets import QStackedWidget, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from lib import utils
from lib import prairie
from gui.mplCanvas import mplCanvas
from lib import diagnostic_tools as dt


class QTabEccentricities(QWidget):

    def __init__(self, in_or_out, parent=None):

        super(QTabEccentricities, self).__init__(parent=None)

        self.main_widget = QStackedWidget(self)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.in_or_out = in_or_out

        main_layout = QVBoxLayout(self.main_widget)
        self.plot = plot(self.main_widget, width=6.5, height=6, dpi=100)
        self.navi_toolbar = NavigationToolbar(self.plot, self)
        main_layout.addWidget(self.navi_toolbar)
        main_layout.addWidget(self.plot)

        self.x_IN_A = np.ones((200, 200))
        self.x_OUT_A = np.ones((200, 200))
        self.y_IN_A = np.ones((200, 200))
        self.y_OUT_A = np.ones((200, 200))

        self.focus = 0
        self.setLayout(main_layout)

    def set_x1(self, x_IN_A):
        self.x_IN_A = x_IN_A

    def set_y1(self, y_IN_A):
        self.y_IN_A = y_IN_A

    def set_x2(self, x_OUT_A):
        self.x_OUT_A = x_OUT_A

    def set_y2(self, y_OUT_A):
        self.y_OUT_A = y_OUT_A

    def set_focus(self, index):
        self.focus = index

    def actualise_ax(self):
        self.plot.fig.clear()
        self.plot.x_IN_A = self.x_IN_A
        self.plot.x_OUT_A = self.x_OUT_A
        self.plot.y_IN_A = self.y_IN_A
        self.plot.y_OUT_A = self.y_OUT_A
        self.plot.in_or_out = self.in_or_out
        self.plot.focus = self.focus
        self.plot.compute_initial_figure()
        self.plot.draw()

    def wait(self):
        pass


class plot(mplCanvas):
    """Simple canvas with a sine plot."""

    def __init__(self, parent, width, height, dpi):

        self.x_IN_A = np.ones((200, 200))
        self.x_OUT_A = np.ones((200, 200))
        self.y_IN_A = np.ones((200, 200))
        self.y_OUT_A = np.ones((200, 200))

        self.ax1 = 0

        self.in_or_out = 'IN'

        self.foc_marker = 0

        self.color = 0

        self.focus = 0

        super(plot, self).__init__(parent, width, height, dpi)

    def compute_initial_figure(self):

        self.fig.clear()

        if self.in_or_out is 'IN':
            self.color = '#018BCF'
        elif self.in_or_out is 'OUT':
            self.color = '#CF2A1B'

        eccentricity = self.y_IN_A
        angular_position_SA = self.x_IN_A

        off = 50

        ref_ecc = eccentricity[0][:]
        ref_ecc = ref_ecc[off:ref_ecc.size - off]
        ref_pos = angular_position_SA[0][:]
        ref_pos = ref_pos[off:ref_pos.size - off]
        ecc_all = []

        def theor_ecc(x, a, b, c):
            return a * np.sin(x + b) + c

        popt, pcov = curve_fit(theor_ecc, ref_pos, ref_ecc, bounds=([-100, -100, -100], [100, 100, 100]))

        for ecc, pos in zip(eccentricity, angular_position_SA):
            ecc = ecc[off:ecc.size - off]
            pos = pos[off:pos.size - off]
            ecc = utils.resample(np.array([pos, ecc]), np.array([ref_pos, ref_ecc]))
            ecc_all.append(ecc[1])

        [ref_ecc, ref_pos] = [eccentricity[0], angular_position_SA[0]]

        deff = []
        residuals_mean = []

        ax1 = self.fig.add_subplot(2, 1, 1)

        ax1.plot(ref_pos, theor_ecc(ref_pos, popt[0], popt[1], popt[2]), linewidth=0.5, color='black')

        for ecc, pos in zip(eccentricity, angular_position_SA):
            ecc = ecc[off:ecc.size - off]
            pos = pos[off:pos.size - off]
            ax1.plot(pos, ecc, linewidth=1)

        ax1.set_title('Position error and eccentricity', loc='left')
        ax1.set_xlabel('Angular position (rad)')
        ax1.set_ylabel('Position error (rad)')
        prairie.style(ax1)

        ax2 = self.fig.add_subplot(2, 2, 3)

        for ecc, pos in zip(eccentricity, angular_position_SA):
            ecc = ecc[off:ecc.size - off]
            pos = pos[off:pos.size - off]
            residuals = ecc - theor_ecc(pos, popt[0], popt[1], popt[2])
            ax2.plot(pos, residuals, linewidth=0.5)
            residuals_mean.append(np.mean(residuals))

        ax2.set_title('Position error after compensation', loc='left')
        ax2.set_xlabel('Angular position (rad)')
        ax2.set_ylabel('Position error (\u03BCrad)')
        prairie.style(ax2)

        ax3 = self.fig.add_subplot(4, 2, 4)

        dt.make_histogram(1e6 * np.asarray(residuals_mean), [-10, 10], '\u03BC' + 'rad', ax3)
        ax3.set_title('Error histogram (' + str(len(eccentricity)) + ' traces)', loc='left')
        ax3.set_ylabel('Occurrence')
        ax3.set_xlabel('Position error (\u03BCrad)')
        prairie.style(ax3)

        self.fig.tight_layout()