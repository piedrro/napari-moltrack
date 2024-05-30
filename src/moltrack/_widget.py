from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
import traceback
from multiprocessing import Manager
from functools import partial
from qtpy.QtCore import QThreadPool

if TYPE_CHECKING:
    import napari

from moltrack.GUI.widget_ui import Ui_Frame as gui

from moltrack.funcs.import_utils import _import_utils
from moltrack.funcs.compute_utils import _compute_utils
from moltrack.funcs.events_utils import _events_utils

subclasses = [_import_utils, _compute_utils, _events_utils]



class QWidget(QWidget, gui, *subclasses):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # create UI
        self.gui = gui()
        self.gui.setupUi(self)

        from moltrack.__init__ import __version__ as version
        print(f"napari-moltrack version: {version}")

        self.initialise_variables()
        self.initialise_events()
        self.initialise_keybindings()

        self.check_gpufit_availibility()

        #create threadpool and stop event
        self.threadpool = QThreadPool()
        manager = Manager()
        self.stop_event = manager.Event()

    def initialise_variables(self):

        self.dataset_dict = {}
        self.localisation_dict = {}
        self.contrast_dict = {}

        self.active_dataset = None

        self.verbose = False

    def initialise_events(self):

        self.gui.import_images.clicked.connect(self.init_import_data)
        self.gui.moltrack_dataset_selector.currentIndexChanged.connect(self.update_active_image)

        self.viewer.dims.events.current_step.connect(self.slider_event)

    def initialise_keybindings(self):


        self.viewer.bind_key('d', self.devfunc)

        pass

    def devfunc(self, viewer=None):

        self.update_active_image()

    def check_gpufit_availibility(self):

        self.gpufit_available = False

        try:
            from pygpufit import gpufit as gf
            package_installed = True
        except:
            package_installed = False

        if package_installed:

            if not gf.cuda_available():
                print("Pygpufit not available due to missing CUDA")
            else:
                runtime_version, driver_version = gf.get_cuda_version()

                runtime_version = ".".join([str(v) for v in list(runtime_version)])
                driver_version = ".".join([str(v) for v in list(driver_version)])

                if runtime_version != driver_version:
                    print(f"Pygpufit not available due to CUDA version mismatch. "
                          f"Runtime: {runtime_version}, Driver: {driver_version}")

                else:
                    self.gpufit_available = True

        else:
            print("Pygpufit not available due to missing package")
            print("Install pygpufit package into napari-PixSeq root directory")

        if self.gpufit_available:
            print("Pygpufit available")
            self.gui.picasso_use_gpufit.setEnabled(True)
            self.gui.picasso_use_gpufit.setChecked(True)
        else:
            self.gui.picasso_use_gpufit.setEnabled(False)


