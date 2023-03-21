
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QRect
import torch as tr


class XYBQtGraphLayoutWidget:
    widget = pg.GraphicsLayoutWidget()
    plot_list = []

    def __init__(self):
        self.widget = pg.GraphicsLayoutWidget()
        self.plot_list = []

    def addPlot(self, name: str):
        plot = self.widget.addPlot(title=name)
        plot_class = XYBQtPlot()
        plot_class.plot = plot
        plot_class.name = name
        self.plot_list.append([name, plot_class])
        return plot_class

    def getPlot(self, name: str):
        for p in self.plot_list:
            if p[0] == name:
                return p[1]
        return None

    def getPlotByIndex(self, index: int):
        return self.plot_list[index][1]

    def getWidget(self):
        return self.widget

class XYBQtPlot:
    name = ''
    plot = None
    item_list = []

    def __init__(self):
        self.name = ''
        self.plot = None
        self.item_list = []

    def addItem(self, name: str, item_type: str):
        item = None
        if item_type == 'PlotDataItem':
            item = pg.PlotDataItem()
        if item_type == 'ScatterPlotItem':
            item = pg.ScatterPlotItem(size=5)
        if item_type == 'ImageItem':
            item = pg.ImageItem()
        self.item_list.append([name, item])
        self.plot.addItem(item)
        return item

    def getItem(self, name=''):
        for i in self.item_list:
            if name == i[0]:
                return i[1]
        return None

    def getItemByIndex(self, index: int):
        return self.item_list[index][1]

