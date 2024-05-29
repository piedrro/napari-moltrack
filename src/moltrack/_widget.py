from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float

if TYPE_CHECKING:
    import napari

from moltrack.GUI.widget_ui import Ui_Frame as gui

class QWidget(QWidget, gui):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # create UI
        self.gui = gui()
        self.gui.setupUi(self)


