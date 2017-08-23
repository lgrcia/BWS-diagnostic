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
from PyQt5.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QGroupBox, QProgressBar, QPushButton, QLabel, QFileDialog, QApplication

from lib import utils
from gui import QTabOPSProcessing
from lib import ops_processing as ops
from gui import QFolderSelectionWidget


class QTabFileProcessing(QWidget):

    def __init__(self, parent=None):

        super(QTabFileProcessing, self).__init__(parent)

        self.parent = parent

        self.superLayout = QHBoxLayout()

        self.actual_TDMS_folder = '...'
        self.actual_destination_folder = '...'

        self.file_box = QGroupBox('Files')
        self.file_box_layout = QVBoxLayout(self)

        self.selection_TDMS = QFolderSelectionWidget.QFolderSelectionWidget('TDMS folder', button=False)
        self.selection_TDMS.label_select_folder.selectionChanged.connect(self.set_TDMS_folder)
        self.file_box_layout.addWidget(self.selection_TDMS)

        self.selection_Destination = QFolderSelectionWidget.QFolderSelectionWidget('Destination folder', button=False)
        self.selection_Destination.label_select_folder.selectionChanged.connect(self.set_destination_folder)
        self.file_box_layout.addWidget(self.selection_Destination)

        self.file_box.setLayout(self.file_box_layout)

        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0, 100)

        self.button_set_parameters = QPushButton('Set parameters', self)
        self.Process = QPushButton('Process', self)
        self.show_processed_data_button = QPushButton('Show PROCESSED data', self)
        self.test_processing = QPushButton('Test Processing', self)
        self.test_processing.clicked.connect(self.test_tdms)
        self.Process.clicked.connect(self.onStart)
        self.button_set_parameters.clicked.connect(self.show_parameters_window)
        self.show_processed_data_button.clicked.connect(self.show_processed_data)

        self.label_progression = QLabel('Waiting for processing')
        self.label_file = QLabel('')

        self.globalLayout = QHBoxLayout(self)

        self.processing_box = QGroupBox('Processing')
        self.processing_box_layout = QVBoxLayout(self)

        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.addWidget(self.file_box, 0, QtCore.Qt.AlignTop)
        processing_action = QHBoxLayout(self)
        processing_action.addWidget(self.button_set_parameters)
        processing_action.addWidget(self.test_processing)

        self.mainLayout.addLayout(processing_action)
        self.mainLayout.addWidget(self.Process)

        self.processing_box_layout.addWidget(self.label_progression)
        self.processing_box_layout.addWidget(self.progressBar)
        self.processing_box_layout.addWidget(self.label_file)
        self.processing_box.setLayout(self.processing_box_layout)

        self.mainLayout.addWidget(self.processing_box, 0, QtCore.Qt.AlignBottom)
        self.mainLayout.addWidget(self.show_processed_data_button)

        self.test_view = QTabOPSProcessing.QTabOPSProcessing(self)

        self.mainLayoutWidget = QWidget(self)
        self.mainLayoutWidget.setLayout(self.mainLayout)

        self.mainLayoutWidget.setFixedHeight(350)
        self.mainLayoutWidget.setFixedWidth(320)

        self.globalLayout.addWidget(self.mainLayoutWidget, 0, QtCore.Qt.AlignTop)
        self.globalLayout.addWidget(self.test_view)

        self.setLayout(self.superLayout)


    def onStart(self):

        #####make security ## ########

        test = utils.tdms_list_from_folder(self.actual_TDMS_folder)

        if type(test) is int:
            self.parent.LogDialog.add('Specified TDMS folder does not contain .tdms files', 'error')

        elif type(utils.extract_from_tdms(self.actual_TDMS_folder + '/' + test[0][0])[0]) is int:
            self.parent.LogDialog.add(
                'TDMS file not loaded because of a key error - try to set [LabView output] in the parameters file',
                'error')

        elif not os.path.exists(self.actual_destination_folder) or self.actual_destination_folder == '...':
            self.parent.LogDialog.add(
                'Please specify a destination folder',
                'error')

        else:

            self.done = False

            self.myLongTask = utils.CreateRawDataFolder(self.actual_TDMS_folder, self.actual_destination_folder, self)
            self.myLongTask.notifyProgress.connect(self.onProgress)
            self.myLongTask.notifyState.connect(self.onState)
            self.myLongTask.notifyFile.connect(self.onFile)
            self.Process.setDisabled(True)
            self.myLongTask.start()

            self.parent.LogDialog.add('Starting ' + self.actual_TDMS_folder + ' conversion', 'info')

    def RAW_IN(self):

        self.myLongTask = ops.ProcessRawData(self.actual_destination_folder + '/RAW_DATA/RAW_IN', self.actual_destination_folder)
        self.myLongTask.notifyProgress.connect(self.onProgress)
        self.myLongTask.notifyState.connect(self.onState)
        self.myLongTask.notifyFile.connect(self.onFile)
        self.Process.setDisabled(True)
        self.myLongTask.start()

    def RAW_OUT(self):

        self.myLongTask = ops.ProcessRawData(self.actual_destination_folder + '/RAW_DATA/RAW_OUT', self.actual_destination_folder)
        self.myLongTask.notifyProgress.connect(self.onProgress)
        self.myLongTask.notifyState.connect(self.onState)
        self.myLongTask.notifyFile.connect(self.onFile)
        self.Process.setDisabled(True)
        self.myLongTask.start()

    def onProgress(self, i):
        self.progressBar.setValue(i)
        if i == 99:
            self.Process.setDisabled(False)
            self.progressBar.reset()

    def onState(self, state):
        print(state)
        self.label_progression.setText(state)
        if state == 'done convert':
            self.Process.setDisabled(False)
            self.progressBar.reset()
            self.RAW_IN()

        elif state == 'done IN':
            self.Process.setDisabled(False)
            self.progressBar.reset()
            self.RAW_OUT()

        elif state == 'done OUT':
            self.Process.setDisabled(False)
            utils.create_processed_data_folder(self.actual_TDMS_folder, destination_folder=self.actual_destination_folder, force_overwrite='y')
            self.progressBar.reset()

    def onFile(self, file):
        self.label_file.setText(file)

    def select_raw_data_folder(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        print(file)
        self.selection_TDMS.label_select_folder.setText(file)

    def show_parameters_window(self):
        os.system('Notepad ' + utils.resource_path('data/parameters.cfg'))

    def set_TDMS_folder(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if file is not '':

            self.parent.LogDialog.add('Selected file : ' + file, 'info')

            self.actual_TDMS_folder = file
            self.selection_TDMS.label_select_folder.setText(file.split('/')[::-1][0])

    def set_destination_folder(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if file is not '':

            self.parent.LogDialog.add('Selected file : ' + file, 'info')

            self.actual_destination_folder = file
            self.selection_Destination.label_select_folder.setText(file.split('/')[::-1][0])

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

    def test_tdms(self):

        test = utils.tdms_list_from_folder(self.actual_TDMS_folder)

        if type(test) is int:
            self.parent.LogDialog.add('Specified TDMS folder does not contain .tdms files', 'error')

        elif type(utils.extract_from_tdms(self.actual_TDMS_folder + '/' + test[0][0])[0]) is int:

            test = utils.extract_from_tdms(self.actual_TDMS_folder + '/' + test[0][0])[0]

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
                self.actual_TDMS_folder + '/' + test[0][0])

            if type(data__s_a_in) is not int:

                self.actualise_single_QTab(self.test_view,
                                           data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out,
                                           t1=time__in, t2=time__out, pd1=data__p_d_in, pd2=data__p_d_out)

                self.parent.LogDialog.add(self.actual_TDMS_folder.split('/')[::-1][0] + '/' + test[0][0] + ' processed', 'info')

            else:

                if data__s_a_in == -1:
                    self.parent.LogDialog.add('TDMS file not loaded because of a key error - try to set [LabView output] in the parameters file', 'error')

                elif data__s_a_in == -2:
                    self.parent.LogDialog.add('One of the range specified is out of data scope - try to set [LabView output] in the parameters file', 'error')

    def show_processed_data(self):

        self.parent.ProcessedAnalysisisTab.set_PROCESSED_folder(self.actual_destination_folder + '/' + self.actual_TDMS_folder.split('/')[::-1][0] + ' PROCESSED')
        self.parent.ProcessedAnalysisisTab.set_TDMS_folder(self.actual_TDMS_folder)
        self.parent.ProcessedAnalysisisTab.actualise_all()
        self.parent.global_tab.setCurrentWidget(self.parent.ProcessedAnalysisisTab)

def main():
    app = QApplication(sys.argv)
    ex = QTabFileProcessing()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()