import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QScrollArea, QVBoxLayout, QMainWindow, QLabel

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        url = ''
        highlight_dir = url + '.'

        self.scrollArea = QScrollArea(widgetResizable=True)
        self.setCentralWidget(self.scrollArea)
        content_widget = QWidget()
        self.scrollArea.setWidget(content_widget)
        lay = QVBoxLayout(content_widget)

        for file in os.listdir(highlight_dir):
            pixmap = QtGui.QPixmap(os.path.join(highlight_dir, file))
            if not pixmap.isNull():
                label = QLabel(pixmap=pixmap)
                lay.addWidget(label)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())