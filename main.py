import numpy as np
import pyqtgraph as pg
import UI.Edit_Main
from PyQt5.QtWidgets import QWidget, QMainWindow
import torch as tr

import pyqtgraph.examples
pg.examples.run()
b = tr.ones(2, 2, 1)
a = tr.ones(2, 2)*2
a = a.reshape(2, 2, 1)
c = tr.cat([a, b], dim=2)

print(c)

app = pg.mkQApp('Test Window')

ui = UI.Edit_Main.Ui_MainWindow()
w = QMainWindow()
ui.setupUi(w)
w.show()
w.setWindowTitle('DeepLearning')
w.resize(800, 400)

app.exec_()
