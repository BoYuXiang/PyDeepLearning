import numpy as np
import pyqtgraph as pg
import UI.Edit_Main
from PyQt5.QtWidgets import QWidget, QMainWindow
import torch as tr


ui = UI.Edit_Main.Ui_MainWindow()
w = QMainWindow()
ui.setupUi(w)
w.show()
w.setWindowTitle('DeepLearning')
w.resize(800, 400)

ui.app.exec_()
