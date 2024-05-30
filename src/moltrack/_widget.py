from typing import TYPE_CHECKING

from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
import traceback
from multiprocessing import Manager
from functools import partial
from qtpy.QtCore import QThreadPool
from qtpy.QtWidgets import QVBoxLayout
import pyqtgraph as pg

if TYPE_CHECKING:
    import napari

from moltrack.GUI.widget_ui import Ui_Frame as gui

from moltrack.funcs.import_utils import _import_utils
from moltrack.funcs.compute_utils import _compute_utils
from moltrack.funcs.events_utils import _events_utils
from moltrack.funcs.segmentation_utils import _segmentation_utils
from moltrack.funcs.picasso_detect_utils import _picasso_detect_utils
from moltrack.funcs.loc_filter_utils import _loc_filter_utils
from moltrack.funcs.picasso_render_utils import _picasso_render_utils

subclasses = [_import_utils, _compute_utils,
              _events_utils, _segmentation_utils,
              _picasso_detect_utils, _loc_filter_utils,
              _picasso_render_utils]

class CustomPyQTGraphWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None


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

        #initialise graph PyQtGraph canvases
        self.gui.filter_graph_container.setLayout(QVBoxLayout())
        self.filter_graph_canvas = CustomPyQTGraphWidget(self)
        self.gui.filter_graph_container.layout().addWidget(self.filter_graph_canvas)

        self.dataset_dict = {}
        self.localisation_dict = {}
        self.contrast_dict = {}

        self.active_dataset = None

        self.verbose = False

    def initialise_events(self):

        self.gui.import_images.clicked.connect(self.init_import_data)
        self.gui.moltrack_dataset_selector.currentIndexChanged.connect(self.update_active_image)

        self.gui.segment_active.clicked.connect(partial(self.initialise_cellpose, mode = "active"))
        self.gui.segment_all.clicked.connect(partial(self.initialise_cellpose, mode = "all"))
        self.gui.cellpose_load_model.clicked.connect(self.load_cellpose_model)
        self.gui.dilate_segmentations.clicked.connect(self.dilate_segmentations)

        self.gui.picasso_detect.clicked.connect(partial(self.init_picasso, detect = True, fit=False))
        self.gui.picasso_fit.clicked.connect(partial(self.init_picasso, detect = False, fit=True))
        self.gui.picasso_detectfit.clicked.connect(partial(self.init_picasso, detect=True, fit=True))

        self.gui.picasso_filter_dataset.currentIndexChanged.connect(self.update_filter_criterion)
        self.gui.filter_criterion.currentIndexChanged.connect(self.update_criterion_ranges)
        self.gui.filter_localisations.clicked.connect(self.pixseq_filter_localisations)
        self.gui.picasso_filter_type.currentIndexChanged.connect(self.update_filter_dataset)

        self.gui.picasso_vis_mode.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_size.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_opacity.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_edge_width.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))

        self.gui.picasso_render.clicked.connect(self.initialise_picasso_render)

        self.viewer.dims.events.current_step.connect(self.slider_event)

    def initialise_keybindings(self):


        self.viewer.bind_key('d', self.devfunc)

        pass

    def devfunc(self, viewer=None):

        self.create_shared_image_chunks()
        self.restore_shared_image_chunks()

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


