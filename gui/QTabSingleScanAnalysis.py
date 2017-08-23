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
import sys

import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QGroupBox, QFileDialog

from lib import utils
from gui import QTabOPSProcessing
from gui import QFolderSelectionWidget


class QTabSingleScanAnalysis(QWidget):

    def __init__(self, parent=None):

        super(QTabSingleScanAnalysis, self).__init__(parent)

        self.parent = parent

        self.superLayout = QHBoxLayout()

        self.actual_TDMS_file = '...'

        self.file_box = QGroupBox('File')
        self.file_box_layout = QVBoxLayout(self)

        self.selection_TDMS = QFolderSelectionWidget.QFolderSelectionWidget('TDMS file', button=False)
        self.selection_TDMS.label_select_folder.selectionChanged.connect(self.set_TDMS_folder)
        self.file_box_layout.addWidget(self.selection_TDMS)

        self.file_box.setLayout(self.file_box_layout)

        self.button_set_parameters = QPushButton('Set parameters', self)
        self.Process = QPushButton('Process', self)

        self.Process.clicked.connect(self.process_tdms)
        self.button_set_parameters.clicked.connect(self.show_parameters_window)

        self.globalLayout = QHBoxLayout(self)

        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.addWidget(self.file_box, 0, QtCore.Qt.AlignTop)
        processing_action = QHBoxLayout(self)
        processing_action.addWidget(self.button_set_parameters)
        processing_action.addWidget(self.Process)

        self.mainLayout.addLayout(processing_action)

        self.process_view = QTabOPSProcessing.QTabOPSProcessing(self)

        self.mainLayoutWidget = QWidget(self)
        self.mainLayoutWidget.setLayout(self.mainLayout)

        self.mainLayoutWidget.setFixedHeight(150)
        self.mainLayoutWidget.setFixedWidth(320)

        self.globalLayout.addWidget(self.mainLayoutWidget, 0, QtCore.Qt.AlignTop)
        self.globalLayout.addWidget(self.process_view)

        self.setLayout(self.superLayout)

    def show_parameters_window(self):
        os.system('Notepad ' + utils.resource_path('data/parameters.cfg'))

    def set_TDMS_folder(self):
        file = QFileDialog.getOpenFileName(self, "Select File")
        file = file[0]

        if file is not '':

            self.parent.LogDialog.add('Selected file : ' + file, 'info')

            self.actual_TDMS_file = file
            self.selection_TDMS.label_select_folder.setText(file.split('/')[::-1][0])

    def actualise_single_QTab(self, QTab, x1, y1, x2, y2, x1_2=None, y1_2=None, x2_2=None, y2_2=None, t1=None, t2=None, pd1=None, pd2=None):

        QTab.set_x_IN_A(x1)
        QTab.set_y_IN_A(y1)
        QTab.set_x_OUT_A(x2)
        QTab.set_y_OUT_A(y2)

        if x1_2 is not None:
            QTab.set_x_IN_B(x1_2)
            QTab.set_y_IN_B(y1_2)
            QTab.set_x_OUT_B(x2_2)
            QTab.set_y_OUT_B(y2_2)

        if t1 is not None:
            QTab.set_t1(t1)
            QTab.set_t2(t2)

        if pd1 is not None:
            QTab.set_pd1(pd1)
            QTab.set_pd2(pd2)

        QTab.actualise_ax()

    def process_tdms(self):

        if not os.path.exists(self.actual_TDMS_file) or self.actual_TDMS_file == '...':
            self.parent.LogDialog.add(
                'Please specify a tdms file to process',
                'error')
        else:

            test = type(utils.extract_from_tdms(self.actual_TDMS_file))

            if test is int:

                if test == -1:
                    self.parent.LogDialog.add(
                        'TDMS file not loaded because of a key error - try to set [LabView output] in the parameters file',
                        'error')

                elif test == -2:
                    self.parent.LogDialog.add(
                        'One of the range specified is out of data scope - try to set [LabView output] in the parameters file',
                        'error')
            else:
                data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
                    self.actual_TDMS_file)

                if type(data__s_a_in) is not int:

                    self.actualise_single_QTab(self.process_view,
                                               data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out,
                                               t1=time__in, t2=time__out, pd1=data__p_d_in, pd2=data__p_d_out)

                    self.parent.LogDialog.add(self.actual_TDMS_file + ' processed', 'info')

                else:
                    if data__s_a_in == -1:
                        self.parent.LogDialog.add('TDMS file not loaded because of a key error - try to set [LabView output] in the parameters file', 'error')

                    elif data__s_a_in == -2:
                        self.parent.LogDialog.add('One of the range specified is out of data scope - try to set [LabView output] in the parameters file', 'error')

def main():
    app = QApplication(sys.argv)
    ex = QTabSingleScanAnalysis()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()