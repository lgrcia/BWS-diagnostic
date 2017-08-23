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
from numpy import arange

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog, QApplication, QTableWidget, QAbstractItemView, QHeaderView, QTableWidgetItem

from gui import QFolderSelectionWidget


def cut(off, data):
    return data[off:data.size - off]


class QMultipleFolderSelection(QWidget):
    def __init__(self, parent=None):
        '''
        QWidget to do a multiple folder selection
        :param parent:
        '''

        super(QMultipleFolderSelection, self).__init__(parent)

        self.parent = parent

        self.folder_selection = QFolderSelectionFrom()
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(0)

        self.plus_minus_layout = QHBoxLayout()

        self.append_button = QPushButton('+')
        self.append_button.setFixedSize(25, 25)
        self.retire_button = QPushButton('-')
        self.retire_button.setFixedSize(25, 25)

        self.plus_minus_layout.addWidget(self.append_button)
        self.plus_minus_layout.addWidget(self.retire_button)

        self.append_all_button = QPushButton('all')
        self.append_all_button.setFixedSize(40, 25)
        self.reset_button = QPushButton('reset')
        self.reset_button.setFixedSize(50, 25)

        self.plus_minus_layout.addWidget(self.append_all_button)
        self.plus_minus_layout.addWidget(self.reset_button)
        self.button_layout.addLayout(self.plus_minus_layout)
        self.plus_minus_layout.setContentsMargins(0, 0, 0, 0)

        self.folder_to = QFolderSelectionTo()

        self.actual_selected_folder_from = ''

        self.mainLayout = QVBoxLayout()

        self.folder_selection.folder_from.label_select_folder.selectionChanged.connect(self.set_actual_folder)
        self.folder_selection.folder_from.button_select_folder.clicked.connect(self.folder_selection.populate)
        self.folder_selection.folder_table_from.clicked.connect(self.set_actual_selected_folder)

        self.append_button.clicked.connect(self.append_folder_to)
        self.retire_button.clicked.connect(self.retire_folder_to)
        self.append_all_button.clicked.connect(self.append_all_folder_to)
        self.reset_button.clicked.connect(self.folder_to.reset)

        self.mainLayout.addWidget(self.folder_selection)
        self.mainLayout.addWidget(self.folder_to)
        self.mainLayout.addLayout(self.button_layout)

        self.setFixedWidth(250)
        self.setFixedHeight(300)

        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.mainLayout)

    def set_actual_folder(self, file=None):

        if file is None:
            file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if file is not '':

            self.folder_selection.set_actual_folder(file)

    def append_folder_to(self):
        self.folder_to.append(self.actual_selected_folder_from)

    def append_all_folder_to(self):
        self.folder_to.subfolder_list = self.folder_selection.subfolder_list
        self.folder_to.actualise()
        self.folder_to.append(self.actual_selected_folder_from)

    def retire_folder_to(self):
        self.folder_to.retire_actual()

    def set_actual_selected_folder(self, index):
        actual_index = index.row()
        self.actual_selected_folder_from = self.folder_selection.subfolder_list[actual_index]

    def get_folder_list(self):
        return self.folder_to.subfolder_list


class QFolderSelectionFrom(QWidget):

    def __init__(self, parent=None):
        super(QFolderSelectionFrom, self).__init__(parent)

        self.parent = parent

        self.mainLayout = QVBoxLayout()

        self.actual_folder = ''
        self.subfolder_list = ''

        self.folder_from = QFolderSelectionWidget.QFolderSelectionWidget('Search in Folder :')
        self.folder_table_from = QTableWidget()

        self.folder_table_from.verticalHeader().hide()
        self.folder_table_from.horizontalHeader().hide()
        self.folder_table_from.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.folder_table_from.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.folder_table_from.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.folder_table_from.horizontalHeader().setFont(QtGui.QFont('Arial', 6))

        self.mainLayout.addWidget(self.folder_from)
        self.mainLayout.addWidget(self.folder_table_from)

        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.mainLayout)

    def return_selected_item_path(self):
        pass

    def populate(self):

        if os.path.exists(self.actual_folder):
            self.subfolder_list = [x[0] for x in os.walk(self.actual_folder)]
            self.subfolder_list = self.subfolder_list[1::]

            self.folder_table_from.setRowCount(len(self.subfolder_list))
            self.folder_table_from.setColumnCount(1)
            self.folder_table_from.horizontalHeader().resizeSection(0, 280)

            for i in arange(0, len(self.subfolder_list)):
                self.folder_table_from.setItem(i, 0, QTableWidgetItem(str(self.subfolder_list[i].split('\\')[::-1][0])))
                self.folder_table_from.item(i, 0).setTextAlignment(QtCore.Qt.AlignVCenter)
                self.folder_table_from.item(i, 0).setFont(QtGui.QFont('Arial', 6))
                self.folder_table_from.verticalHeader().resizeSection(i, 20)

        else:
            pass

    def set_actual_folder(self, folder):

        self.actual_folder = folder
        self.folder_from.label_select_folder.setText(folder.split('/')[::-1][0])


class QFolderSelectionTo(QWidget):

    def __init__(self, parent=None):
        super(QFolderSelectionTo, self).__init__(parent)

        self.parent = parent

        self.mainLayout = QVBoxLayout()

        self.actual_selected_fodler = 0

        self.subfolder_list = []

        self.folder_table_to = QTableWidget()

        self.folder_table_to.clicked.connect(self.set_actual_selected_folder)

        self.folder_table_to.verticalHeader().hide()
        self.folder_table_to.horizontalHeader().hide()
        self.folder_table_to.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.folder_table_to.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.folder_table_to.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.folder_table_to.horizontalHeader().setFont(QtGui.QFont('Arial', 6))

        self.mainLayout.addWidget(self.folder_table_to)

        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.mainLayout)

    def return_selected_item_path(self):
        pass

    def append(self, folder):

        if folder not in self.subfolder_list:

            self.subfolder_list.append(folder)

            self.actualise()

    def reset(self):

        self.folder_table_to.reset()

    def set_actual_selected_folder(self, index):

        self.actual_selected_fodler = index.row()

    def actualise(self):

        self.folder_table_to.setRowCount(len(self.subfolder_list))
        self.folder_table_to.setColumnCount(1)
        self.folder_table_to.horizontalHeader().resizeSection(0, 280)

        for i in arange(0, len(self.subfolder_list)):
            self.folder_table_to.setItem(i, 0,
                                         QTableWidgetItem(str(self.subfolder_list[i].split('\\')[::-1][0])))
            self.folder_table_to.item(i, 0).setTextAlignment(QtCore.Qt.AlignVCenter)
            self.folder_table_to.item(i, 0).setFont(QtGui.QFont('Arial', 6))
            self.folder_table_to.verticalHeader().resizeSection(i, 20)

    def retire_actual(self):

        if self.actual_selected_fodler < len(self.subfolder_list):
            self.subfolder_list.pop(self.actual_selected_fodler)
            self.actualise()

    def reset(self):

        self.actual_selected_fodler = 0
        self.subfolder_list = []
        self.folder_table_to.reset()
        self.actualise()


def main():
    app = QApplication(sys.argv)
    ex = QMultipleFolderSelection()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()








