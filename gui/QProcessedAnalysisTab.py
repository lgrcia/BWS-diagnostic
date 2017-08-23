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
import sys
import time
import shutil
import numpy as np
import configparser
import scipy.io as sio

from numpy import arange
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QTableWidgetItem, QFileDialog, QVBoxLayout, QHBoxLayout

from lib import utils
from gui import QTabWidgetPlotting
from gui import QFileDescriptionTable
from gui import QCalibrationInformation
from gui import Calibration


def cut(off, data):
    return data[off:data.size-off]


class QProcessedAnalysisTab(QWidget):

    def __init__(self, parent=None):

        super(QProcessedAnalysisTab, self).__init__(parent)

        self.parent = parent

        # Window properties
        self.setWindowTitle('OPS Processing')

        self.actual_index = 0
        # self.actual_PROCESSED_folder = "F:\pyBWS01\data/133rs_CC__2017_06_06__17_08 PROCESSED"
        # self.actual_TDMS_folder = "E:\BWS_Fatigue_Tests\Calibration_Bench_Files/133rs_CC__2017_06_06__17_08"
        self.actual_PROCESSED_folder = "Click here to select"
        self.actual_TDMS_folder = "Click here to select"
        # self.tdms_file_list = utils.tdms_list_from_folder(self.actual_TDMS_folder)
        self.tdms_file_list = 0

        self.TabWidgetPlotting = QTabWidgetPlotting.QTabWidgetPlotting()
        self.FileDescriptionTable = QFileDescriptionTable.QFileDescriptionTable()
        self.CalibrationInformation = QCalibrationInformation.QCalibrationInformation(parent=self)
        # self.Multiple_folder_selection = QMultipleFolderSelection.QMultipleFolderSelection()
        self.mainLayout = QVBoxLayout

        self.FileDescriptionTable.see_raw_button.clicked.connect(self.select_index_tdms)
        self.FileDescriptionTable.set_parameters_button.clicked.connect(self.show_parameters_window)
        self.FileDescriptionTable.dump_button.clicked.connect(self.dump_actual_scan)

        self.CalibrationInformation.processed_data_selection.label_select_folder.selectionChanged.connect(self.set_PROCESSED_folder)
        self.CalibrationInformation.tdms_data_selection.label_select_folder.selectionChanged.connect(self.set_TDMS_folder)
        self.CalibrationInformation.processed_data_selection.button_select_folder.pressed.connect(self.actualise_all)
        self.FileDescriptionTable.table.clicked.connect(self.select_index)

        self.TabWidgetPlotting.tab_eccentricity.actualise_ax()

        self.mainLayout = QVBoxLayout()

        self.secondLayout = QHBoxLayout()
        self.secondLayout.addWidget(self.CalibrationInformation, 0, QtCore.Qt.AlignTop)
        self.secondLayout.addWidget(self.TabWidgetPlotting)
        self.secondLayout.addWidget(self.FileDescriptionTable, 0, QtCore.Qt.AlignRight)

        self.mainLayout.addLayout(self.secondLayout)

        self.calibration = 0

        #######################
        #
        # self.CalibrationInformation.set_PROCESSED_folder(self.actual_PROCESSED_folder)
        # self.actual_PROCESSED_folder = self.actual_PROCESSED_folder
        # self.actual_index = 0
        # self.calibration = Calibration.Calibration(self.actual_PROCESSED_folder)
        #
        # self.CalibrationInformation.set_TDMS_folder(self.actual_TDMS_folder)
        # self.actual_TDMS_folder = self.actual_TDMS_folder
        # self.actual_index = 0
        # self.tdms_file_list = utils.tdms_list_from_folder(self.actual_TDMS_folder)

        ########################~


        self.setLayout(self.mainLayout)

        # Window properties
        self.resize(100, 50)

    def select_index(self, index):

        self.actual_index = index.row()
        self.actualise_not_folder_dependant_plot()
        self.TabWidgetPlotting.tab_OPS_processing.reset()

        self.parent.LogDialog.add(self.actual_PROCESSED_folder.split('/')[::-1][0] + ' - scan at index ' + str(self.actual_index) +  ' loaded', 'info')

    def actualise_all(self):

        info_set_bool = self.CalibrationInformation.set_PROCESSED_folder(self.actual_PROCESSED_folder)

        if info_set_bool == -1:
            self.parent.LogDialog.add('PROCESSED info not found in this folder', 'error')

        else:

            self.calibration = Calibration.Calibration(self.actual_PROCESSED_folder)

            self.tdms_file_list = utils.tdms_list_from_folder(self.actual_TDMS_folder)

            self.actualise_file_table()

            self.actualise_single_QTab(self.TabWidgetPlotting.tab_position,
                                       x1=self.calibration.time_IN_SA[self.actual_index],
                                       y1=self.calibration.angular_position_SA_IN[self.actual_index],
                                       x2=self.calibration.time_OUT_SA[self.actual_index],
                                       y2=self.calibration.angular_position_SA_OUT[self.actual_index],
                                       x1_2=self.calibration.time_IN_SB[self.actual_index],
                                       y1_2=self.calibration.angular_position_SB_IN[self.actual_index],
                                       x2_2=self.calibration.time_OUT_SB[self.actual_index],
                                       y2_2=self.calibration.angular_position_SB_OUT[self.actual_index])

            self.actualise_single_QTab(self.TabWidgetPlotting.tab_speed,
                                       x1=cut(2, self.calibration.time_IN_SA[self.actual_index][
                                                 0:self.calibration.time_IN_SA[self.actual_index].size - 1]),
                                       y1=cut(2, self.calibration.speed_IN_SA[self.actual_index]),
                                       x2=cut(2, self.calibration.time_OUT_SA[self.actual_index][
                                                 0:self.calibration.time_OUT_SA[self.actual_index].size - 1]),
                                       y2=cut(2, self.calibration.speed_OUT_SA[self.actual_index]),
                                       x1_2=cut(2, self.calibration.time_IN_SB[self.actual_index][
                                                   0:self.calibration.time_IN_SB[self.actual_index].size - 1]),
                                       y1_2=cut(2, self.calibration.speed_IN_SB[self.actual_index]),
                                       x2_2=cut(2, self.calibration.time_OUT_SB[self.actual_index][
                                                   0:self.calibration.time_OUT_SB[self.actual_index].size - 1]),
                                       y2_2=cut(2, self.calibration.speed_OUT_SB[self.actual_index]))

            self.actualise_single_QTab(self.TabWidgetPlotting.tab_calibration_IN,
                                       self.calibration.occlusion_IN,
                                       self.calibration.laser_position_IN, 0, 0)

            self.actualise_single_QTab(self.TabWidgetPlotting.tab_calibration_OUT,
                                       self.calibration.occlusion_OUT,
                                       self.calibration.laser_position_OUT, 0, 0)

            self.actualise_single_QTab(self.TabWidgetPlotting.tab_RDS,
                                       self.calibration.time_IN_SA,
                                       self.calibration.time_OUT_SA,
                                       self.calibration.time_IN_SB,
                                       self.calibration.time_OUT_SB)

            self.actualise_single_QTab(self.TabWidgetPlotting.tab_eccentricity,
                                       x1=cut(20, self.calibration.angular_position_SA_IN[self.actual_index]),
                                       y1=cut(20, self.calibration.eccentricity_IN[self.actual_index]),
                                       x2=cut(20, self.calibration.angular_position_SA_OUT[self.actual_index]),
                                       y2=cut(20, self.calibration.eccentricity_OUT[self.actual_index]),
                                       x1_2=0, y1_2=0, x2_2=0, y2_2=0)

            self.FileDescriptionTable.table.setHorizontalHeaderLabels(['Laser pos\n(LabView)',
                                                                       'Laser pos\n(Absolute)',
                                                                       'Scan\nnumber', 'occlusion\nIN (rad)',
                                                                       'occlusion\nOUT (rad)'])

            self.TabWidgetPlotting.setCurrentWidget(self.TabWidgetPlotting.tab_calibration_IN)

            self.TabWidgetPlotting.tab_OPS_processing.reset()

            self.parent.LogDialog.add('PROCESSED data imported', 'info')

    def actualise_not_folder_dependant_plot(self):

        self.actualise_single_QTab(self.TabWidgetPlotting.tab_position,
                                   x1=self.calibration.time_IN_SA[self.actual_index],
                                   y1=self.calibration.angular_position_SA_IN[self.actual_index],
                                   x2=self.calibration.time_OUT_SA[self.actual_index],
                                   y2=self.calibration.angular_position_SA_OUT[self.actual_index],
                                   x1_2=self.calibration.time_IN_SB[self.actual_index],
                                   y1_2=self.calibration.angular_position_SB_IN[self.actual_index],
                                   x2_2=self.calibration.time_OUT_SB[self.actual_index],
                                   y2_2=self.calibration.angular_position_SB_OUT[self.actual_index])

        self.actualise_single_QTab(self.TabWidgetPlotting.tab_speed,
                                   x1=cut(2, self.calibration.time_IN_SA[self.actual_index][
                                             0:self.calibration.time_IN_SA[self.actual_index].size - 1]),
                                   y1=cut(2, self.calibration.speed_IN_SA[self.actual_index]),
                                   x2=cut(2, self.calibration.time_OUT_SA[self.actual_index][
                                             0:self.calibration.time_OUT_SA[self.actual_index].size - 1]),
                                   y2=cut(2, self.calibration.speed_OUT_SA[self.actual_index]),
                                   x1_2=cut(2, self.calibration.time_IN_SB[self.actual_index][
                                               0:self.calibration.time_IN_SB[self.actual_index].size - 1]),
                                   y1_2=cut(2, self.calibration.speed_IN_SB[self.actual_index]),
                                   x2_2=cut(2, self.calibration.time_OUT_SB[self.actual_index][
                                               0:self.calibration.time_OUT_SB[self.actual_index].size - 1]),
                                   y2_2=cut(2, self.calibration.speed_OUT_SB[self.actual_index]))

        self.TabWidgetPlotting.tab_calibration_IN.set_focus(self.actual_index)
        self.TabWidgetPlotting.tab_calibration_OUT.set_focus(self.actual_index)

        self.actualise_single_QTab(self.TabWidgetPlotting.tab_eccentricity,
                                   x1=cut(20, self.calibration.angular_position_SA_IN[self.actual_index]),
                                   y1=cut(20, self.calibration.eccentricity_IN[self.actual_index]),
                                   x2=cut(20, self.calibration.angular_position_SA_OUT[self.actual_index]),
                                   y2=cut(20, self.calibration.eccentricity_OUT[self.actual_index]),
                                   x1_2=0, y1_2=0, x2_2=0, y2_2=0)

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

    def actualise_file_table(self):

        folder = self.actual_PROCESSED_folder

        parameter_file = utils.resource_path('data/parameters.cfg')
        config = configparser.RawConfigParser()
        config.read(parameter_file)
        tank_center = eval(config.get('OPS processing parameters', 'offset_center'))

        data = sio.loadmat(folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)
        laser_position = data['laser_position']
        scan_number = data['scan_number']
        occlusion_IN = data['occlusion_position']
        data = sio.loadmat(folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)
        occlusion_OUT = data['occlusion_position']

        self.FileDescriptionTable.table.setRowCount(laser_position.size)
        self.FileDescriptionTable.table.setColumnCount(5)

        self.FileDescriptionTable.table.horizontalHeader().resizeSection(0, 55)
        self.FileDescriptionTable.table.horizontalHeader().resizeSection(1, 55)
        self.FileDescriptionTable.table.horizontalHeader().resizeSection(2, 40)
        self.FileDescriptionTable.table.horizontalHeader().resizeSection(3, 55)
        self.FileDescriptionTable.table.horizontalHeader().resizeSection(4, 55)

        font2 = QtGui.QFont()
        font2.setPointSize(7)

        # self.FileDescriptionTable.table.setFont(font2)

        font = QtGui.QFont()
        font.setPointSize(8)

        for i in arange(0, laser_position.size):
            self.FileDescriptionTable.table.setItem(i, 0, QTableWidgetItem(str(laser_position[i])))
            self.FileDescriptionTable.table.setItem(i, 1, QTableWidgetItem(str( - laser_position[i] + tank_center )))
            self.FileDescriptionTable.table.setItem(i, 2, QTableWidgetItem(str(scan_number[i])))
            self.FileDescriptionTable.table.setItem(i, 3, QTableWidgetItem(str(occlusion_IN[i])))
            self.FileDescriptionTable.table.setItem(i, 4, QTableWidgetItem(str(occlusion_OUT[i])))
            self.FileDescriptionTable.table.item(i, 0).setTextAlignment(QtCore.Qt.AlignCenter)
            self.FileDescriptionTable.table.item(i, 1).setTextAlignment(QtCore.Qt.AlignCenter)
            self.FileDescriptionTable.table.item(i, 2).setTextAlignment(QtCore.Qt.AlignCenter)
            self.FileDescriptionTable.table.item(i, 3).setTextAlignment(QtCore.Qt.AlignCenter)
            self.FileDescriptionTable.table.item(i, 4).setTextAlignment(QtCore.Qt.AlignCenter)

            self.FileDescriptionTable.table.item(i, 0).setFont(font)
            self.FileDescriptionTable.table.item(i, 1).setFont(font)
            self.FileDescriptionTable.table.item(i, 2).setFont(font)
            self.FileDescriptionTable.table.item(i, 3).setFont(font)
            self.FileDescriptionTable.table.item(i, 4).setFont(font)

    def set_PROCESSED_folder(self, file=None):

        if file is None:
            file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if file is not '':

            self.parent.LogDialog.add('Selected file : ' + file, 'info')

            info_set_bool = self.CalibrationInformation.set_PROCESSED_folder(file)

            if info_set_bool == -1:
                self.parent.LogDialog.add('PROCESSED info not found in this folder', 'warning')

            else:

                self.actual_PROCESSED_folder = file
                self.actual_index = 0

                if self.actual_PROCESSED_folder.find(self.actual_TDMS_folder.split('/')[::-1][0]) == -1:
                    self.parent.LogDialog.add('TDMS folder name and PROCESSED folder name do not match', 'warning')

                self.calibration = Calibration.Calibration(self.actual_PROCESSED_folder)

    def set_TDMS_folder(self, file=None):

        if file is None:
            file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if file is not '':

            self.parent.LogDialog.add('Selected file : ' + file, 'info')

            self.actual_TDMS_folder = file
            self.tdms_file_list = utils.tdms_list_from_folder(file)
            self.CalibrationInformation.set_TDMS_folder(file)

            if self.actual_PROCESSED_folder.find(self.actual_TDMS_folder.split('/')[::-1][0]) == -1:
                self.parent.LogDialog.add('TDMS folder name and PROCESSED folder name do not match', 'warning')

            else:
                self.parent.LogDialog.add('TDMS folder name and PROCESSED folder name are matching', 'info')

    def select_index_tdms(self):

        self.tdms_file_list = utils.tdms_list_from_folder(self.actual_TDMS_folder)

        if type(self.tdms_file_list) is int:
            self.parent.LogDialog.add('Specified TDMS folder does not contain .tdms files', 'error')

        else:
            data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out, data__p_d_in, data__p_d_out, time__in, time__out = utils.extract_from_tdms(
                self.actual_TDMS_folder + '/' + self.tdms_file_list[0][self.actual_index])

            if type(data__s_a_in) is not int:

                self.actualise_single_QTab(self.TabWidgetPlotting.tab_OPS_processing,
                                           data__s_a_in, data__s_b_in, data__s_a_out, data__s_b_out,
                                           t1=time__in, t2=time__out, pd1=data__p_d_in, pd2=data__p_d_out)

                self.TabWidgetPlotting.setCurrentWidget(self.TabWidgetPlotting.tab_OPS_processing)
                self.FileDescriptionTable.table.selectRow(self.actual_index)

                self.parent.LogDialog.add(self.actual_TDMS_folder.split('/')[::-1][0] + '/' + self.tdms_file_list[0][self.actual_index] + ' processed', 'info')

            else:
                self.parent.LogDialog.add('TDMS file not loaded because of a key error - try to set [LabView output] in the parameters file', 'error')

    def dump_actual_scan(self):

        final_dictionary = {}

        matfile = sio.loadmat(self.actual_PROCESSED_folder + '/PROCESSED_IN.mat', struct_as_record=False, squeeze_me=True)
        keys = list(matfile.keys())[3::]

        for key in keys:
            data = matfile[key]
            if len(data) != 0:
                data = np.delete(data, self.actual_index)

            final_dictionary[key] = data

        sio.savemat(self.actual_PROCESSED_folder + '/PROCESSED_IN.mat', final_dictionary)

        final_dictionary = {}

        matfile = sio.loadmat(self.actual_PROCESSED_folder + '/PROCESSED_OUT.mat', struct_as_record=False,
                              squeeze_me=True)
        keys = list(matfile.keys())[3::]

        for key in keys:
            data = matfile[key]
            if len(data) != 0:
                data = np.delete(data, self.actual_index)

            final_dictionary[key] = data

        sio.savemat(self.actual_PROCESSED_folder + '/PROCESSED_OUT.mat', final_dictionary)

        if not os.path.exists(self.actual_TDMS_folder + '/DUMPED'):
            os.makedirs(self.actual_TDMS_folder + '/DUMPED')

        shutil.move(self.actual_TDMS_folder + '/' + self.tdms_file_list[0][self.actual_index], self.actual_TDMS_folder + '/DUMPED')

        self.CalibrationInformation.set_PROCESSED_folder(self.actual_PROCESSED_folder)
        self.CalibrationInformation.set_TDMS_folder(self.actual_TDMS_folder)
        self.actualise_all()

    def show_parameters_window(self):
        self.parent.LogDialog.add('Opening ' + utils.resource_path('data/parameters.cfg') + ' ...', 'info')
        time.sleep(2)
        os.system('Notepad ' + utils.resource_path('data/parameters.cfg'))


def main():
    app = QApplication(sys.argv)
    ex = QProcessedAnalysisTab()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()








