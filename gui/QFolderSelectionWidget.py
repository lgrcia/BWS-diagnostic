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

from PyQt5.QtWidgets import QWidget, QTextEdit, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5 import QtGui, QtCore


class QFolderSelectionWidget(QWidget):

    def __init__(self, Name, default_fodler=None, button=True, parent=None):
        '''
        Widget to specify a folder
        :param Name: Name of the section
        :param default_fodler: folder to be displayed by default
        :param button: boolean to show a button or not
        :param parent:
        '''

        super(QFolderSelectionWidget, self).__init__(parent)

        self.Name = Name
        self.folder = 'Click here to select'

        self.label_name = QLabel(Name)

        if button is True:
            self.button_select_folder = QPushButton('Import', self)
            self.button_select_folder.setFixedWidth(60)

            self.label_select_folder = QTextEdit(self.folder)
            self.label_select_folder.setFixedSize(150, 30)
            self.label_select_folder.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            font = QtGui.QFont()
            font.setPointSize(7)
            self.label_select_folder.setFont(font)

            self.mainLayout = QVBoxLayout(self)
            self.mainLayout.setContentsMargins(0, 0, 0, 0)
            select_folder = QHBoxLayout(self)
            select_folder.addWidget(self.label_select_folder, 0, QtCore.Qt.AlignLeft)
            select_folder.addWidget(self.button_select_folder)
            self.mainLayout.addWidget(self.label_name)
            self.mainLayout.addLayout(select_folder)


        else:

            self.label_select_folder = QTextEdit(self.folder)
            self.label_select_folder.setFixedSize(210, 30)
            self.label_select_folder.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            font = QtGui.QFont()
            font.setPointSize(7)
            self.label_select_folder.setFont(font)

            self.mainLayout = QVBoxLayout(self)
            self.mainLayout.setContentsMargins(0, 0, 0, 0)
            select_folder = QHBoxLayout(self)
            select_folder.addWidget(self.label_select_folder, 0, QtCore.Qt.AlignLeft)
            self.mainLayout.addWidget(self.label_name)
            self.mainLayout.addLayout(select_folder)

        self.setFixedHeight(60)

    def set_folder(self, folder):
        self.folder = folder
        label = folder.split('/')[::-1][0]
        self.label_select_folder.setText(label)

