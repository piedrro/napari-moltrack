from functools import partial
from multiprocessing import Manager
from typing import TYPE_CHECKING

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph import ImageView
import tifffile
from qtpy.QtCore import QThreadPool
from qtpy.QtWidgets import QVBoxLayout, QWidget
from skimage import exposure
import numpy as np
import torch
import traceback
from napari.utils.notifications import show_info
from PyQt5.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QVBoxLayout, QWidget, QMainWindow
from PyQt5 import uic

if TYPE_CHECKING:
    import napari

from moltrack.funcs.compute_utils import _compute_utils
from moltrack.funcs.events_utils import _events_utils
from moltrack.funcs.export_utils import _export_utils
from moltrack.funcs.import_utils import _import_utils
from moltrack.funcs.loc_filter_utils import _loc_filter_utils
from moltrack.funcs.track_filter_utils import _track_filter_utils
from moltrack.funcs.picasso_detect_utils import _picasso_detect_utils
from moltrack.funcs.picasso_render_utils import _picasso_render_utils
from moltrack.funcs.segmentation_events import _segmentation_events
from moltrack.funcs.segmentation_utils import _segmentation_utils
from moltrack.funcs.tracking_utils import _tracking_utils
from moltrack.funcs.bactfit_utils import _bactfit_utils
from moltrack.funcs.cell_events import _cell_events
from moltrack.funcs.oufti_utils import oufti
from moltrack.funcs.diffusion_utils import _diffusion_utils
from moltrack.funcs.cell_heatmap_utils import _cell_heatmap_utils
from moltrack.funcs.traces_utils import _traces_utils

from moltrack.GUI.widget_ui import Ui_Frame as gui

subclasses = [_import_utils, _compute_utils,
              _events_utils, _segmentation_utils,
              _picasso_detect_utils, _loc_filter_utils,
              _picasso_render_utils, _tracking_utils,
              _export_utils, _segmentation_events,
              _bactfit_utils, _cell_events,
              oufti, _diffusion_utils, _cell_heatmap_utils,
              _track_filter_utils, _traces_utils]

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

        show_info(f"napari-moltrack version: {version}")

        self.initialise_variables()
        self.initialise_events()
        self.initialise_keybindings()

        self.check_cuda_availibility()
        self.check_gpufit_availibility()
        self.update_detect_options()
        self.initialise_channel_selectors()
        self.update_import_options()
        self.update_heatmap_options()
        self.update_diffusion_options()
        self.update_locs_import_options()

        # create threadpool and stop event
        self.threadpool = QThreadPool()
        manager = Manager()
        self.stop_event = manager.Event()

    def initialise_variables(self):
        # initialise graph PyQtGraph canvases
        self.gui.filter_graph_container.setLayout(QVBoxLayout())
        self.filter_graph_canvas = CustomPyQTGraphWidget(self)
        self.gui.filter_graph_container.layout().addWidget(self.filter_graph_canvas)

        self.gui.tracks_filter_graph_container.setLayout(QVBoxLayout())
        self.track_graph_canvas = CustomPyQTGraphWidget(self)
        self.gui.tracks_filter_graph_container.layout().addWidget(self.track_graph_canvas)

        self.gui.adc_graph_container.setLayout(QVBoxLayout())
        self.adc_graph_canvas = CustomPyQTGraphWidget(self)
        self.gui.adc_graph_container.layout().addWidget(self.adc_graph_canvas)

        self.heatmap_canvas = ImageView()
        self.gui.heatmap_graph_container.setLayout(QVBoxLayout())
        self.gui.heatmap_graph_container.layout().addWidget(self.heatmap_canvas)
        self.heatmap_canvas.ui.histogram.hide()
        self.heatmap_canvas.ui.roiBtn.hide()
        self.heatmap_canvas.ui.menuBtn.hide()

        self.dataset_dict = {}
        self.localisation_dict = {}
        self.tracking_dict = {}
        self.contrast_dict = {}
        self.diffusion_dict = {}

        self.active_dataset = None
        self.active_channel = None

        self.verbose = False

        self.segmentation_mode = "panzoom"
        self.interface_mode = "segment"


    def initialise_events(self):

        self.gui.import_images.clicked.connect(self.init_import_data)

        self.gui.import_mode.currentIndexChanged.connect(self.update_import_options)
        self.gui.import_multichannel_mode.currentIndexChanged.connect(self.update_import_options)

        self.gui.moltrack_dataset_selector.currentIndexChanged.connect(self.update_active_image)
        self.gui.moltrack_channel_selector.currentIndexChanged.connect(self.update_active_image)

        self.gui.segment_active.clicked.connect(partial(self.initialise_cellpose, mode="active"))
        self.gui.cellpose_load_model.clicked.connect(self.load_cellpose_model)
        self.gui.dilate_segmentations.clicked.connect(self.dilate_segmentations)

        self.gui.smlm_detect_mode.currentIndexChanged.connect(self.update_detect_options)

        self.gui.picasso_detect.clicked.connect(partial(self.init_picasso, detect=True, fit=False))
        self.gui.picasso_fit.clicked.connect(partial(self.init_picasso, detect=False, fit=True))
        self.gui.picasso_detectfit.clicked.connect(partial(self.init_picasso, detect=True, fit=True))

        self.gui.picasso_filter_dataset.currentIndexChanged.connect(self.update_filter_criterion)
        self.gui.filter_criterion.currentIndexChanged.connect(self.update_criterion_ranges)
        self.gui.filter_subtract_bg.stateChanged.connect(self.update_criterion_ranges)
        self.gui.filter_localisations.clicked.connect(self.pixseq_filter_localisations)

        self.gui.track_filter_dataset.currentIndexChanged.connect(self.update_track_filter_criterion)
        self.gui.track_filter_channel.currentIndexChanged.connect(self.update_track_filter_criterion)
        self.gui.track_filter_criterion.currentIndexChanged.connect(self.update_track_filter_metric)
        self.gui.track_filter_metric.currentIndexChanged.connect(self.update_track_criterion_ranges)
        self.gui.track_filter_subtract_bg.stateChanged.connect(self.update_track_criterion_ranges)
        self.gui.filter_tracks.clicked.connect(self.filter_tracks)
        self.gui.compute_track_stats.clicked.connect(self.initialise_track_stats)

        self.gui.compute_pixmap.clicked.connect(self.initialise_pixmap)

        self.gui.picasso_segmentation_layer.currentIndexChanged.connect(self.update_picasso_segmentation_filter)

        self.gui.picasso_vis_mode.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_size.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_opacity.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_edge_width.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))

        self.gui.picasso_render.clicked.connect(self.initialise_picasso_render)

        self.gui.link_localisations.clicked.connect(self.initialise_tracking)

        self.gui.export_localisations.clicked.connect(self.initialise_export_locs)

        self.gui.fit_segmentations.clicked.connect(self.initialise_bactfit)

        self.gui.shapes_export_data.currentIndexChanged.connect(self.update_shape_export_options)
        self.gui.export_shapes.clicked.connect(self.init_export_shapes_data)

        self.gui.locs_export_data.currentIndexChanged.connect(self.update_locs_export_options)

        self.gui.remove_seglocs.clicked.connect(self.remove_seglocs)
        self.gui.remove_segtracks.clicked.connect(self.remove_segtracks)
        self.gui.segtracks_detect.currentIndexChanged.connect(self.update_segtrack_options)

        self.gui.adc_plot.currentIndexChanged.connect(self.plot_diffusion_graph)
        self.gui.adc_channel.currentIndexChanged.connect(self.plot_diffusion_graph)
        self.gui.adc_dataset.currentIndexChanged.connect(self.plot_diffusion_graph)
        self.gui.adc_range_min.valueChanged.connect(self.plot_diffusion_graph)
        self.gui.adc_range_max.valueChanged.connect(self.plot_diffusion_graph)
        self.gui.adc_bins.valueChanged.connect(self.plot_diffusion_graph)
        self.gui.adc_density.stateChanged.connect(self.plot_diffusion_graph)
        self.gui.adc_hide_first.stateChanged.connect(self.plot_diffusion_graph)
        self.gui.export_adc.clicked.connect(self.export_diffusion_graph)

        self.gui.adc_plot.currentIndexChanged.connect(self.update_diffusion_options)

        self.gui.show_data.stateChanged.connect(partial(self.update_active_layers, mode="data"))
        self.gui.show_shapes.stateChanged.connect(partial(self.update_active_layers, mode="shapes"))
        self.gui.show_tracks.stateChanged.connect(partial(self.update_active_layers, mode="tracks"))
        self.gui.show_locs.stateChanged.connect(partial(self.update_active_layers, mode="locs"))
        self.gui.show_render.stateChanged.connect(partial(self.update_active_layers, mode="render"))

        self.gui.compute_heatmap.clicked.connect(self.compute_cell_heatmap)
        self.gui.export_heatmap.clicked.connect(self.export_cell_heatmap)
        self.gui.heatmap_mode.currentIndexChanged.connect(self.update_heatmap_options)
        self.gui.generate_heatmap.clicked.connect(self.plot_heatmap)
        self.gui.heatmap_length_reset.clicked.connect(self.update_render_length_range)
        self.gui.heatmap_msd_reset.clicked.connect(self.update_render_msd_range)

        self.gui.export_traces.clicked.connect(self.export_traces)

        self.viewer.layers.events.inserted.connect(self.update_layer_combos)
        self.viewer.layers.events.removed.connect(self.update_layer_combos)
        self.viewer.dims.events.current_step.connect(self.slider_event)

        self.gui.traces_export_dataset.currentIndexChanged.connect(self.update_traces_export_options)
        self.gui.traces_export_channel.currentIndexChanged.connect(self.update_traces_export_options)

        self.gui.import_localisations.clicked.connect(self.import_localisations)
        self.gui.locs_import_data.currentIndexChanged.connect(self.update_locs_import_options)

    def devfunc(self, viewer=None):

        # self.update_render_length_range()
        # self.update_render_msd_range()
        self.update_ui()
        self.draw_localisations()

        # self.update_traces_export_options()

        # self.update_pixmap_options()
        # self.compute_pixmap_finished()
        # self.celllist.get_cell_lengths()
        # self.update_render_length_range()

        # self.update_ui()
        # self.plot_cell_heatmap()
        # self.plot_cell_render()
        # print(True)
        # self.tracking_dict = {}

        # self.populate_dataset_selectors()

        # self.create_shared_image_chunks()
        # self.restore_shared_image_chunks()
        # self.update_traces_export_options()
        self.gui.track_filter_channel.currentIndexChanged.connect(self.update_track_filter_criterion)

    def initialise_keybindings(self):

        self.viewer.bind_key("F1", self.devfunc)

        self.viewer.bind_key(key="Control-Right", func=lambda event: self.moltract_translation(direction="right"), overwrite=True, )
        self.viewer.bind_key(key="Control-Left", func=lambda event: self.moltract_translation(direction="left"), overwrite=True, )
        self.viewer.bind_key(key="Control-Up", func=lambda event: self.moltract_translation(direction="up"), overwrite=True, )
        self.viewer.bind_key(key="Control-Down", func=lambda event: self.moltract_translation(direction="down"), overwrite=True, )
        self.viewer.bind_key(key="Control-Z", func=self.moltrack_undo, overwrite=True, )

        self.register_shape_layer_keybinds(self.viewer)

    def remove_keybindings(self):

        self.viewer.bind_key("F1", None)

        self.viewer.bind_key(key="Control-Right", func=None)
        self.viewer.bind_key(key="Control-Left", func=None)
        self.viewer.bind_key(key="Control-Up", func=None)
        self.viewer.bind_key(key="Control-Down", func=None)
        self.viewer.bind_key(key="Control-Z", func=None)

        self.remove_shape_layer_keybinds(self.viewer)

    def normalize99(self, X):
        """ normalize image so 0.0==0.01st percentile and 1.0==99.99th percentile """
        X = X.copy()

        if np.max(X) > 0:
            v_min, v_max = np.percentile(X[X != 0], (0.01, 99.99))
            X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

        return X

    def rescale01(self, x):
        """ normalize image from 0 to 1 """

        if np.max(x) > 0:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))

        x = x.astype(np.float64)

        return x

    def check_cuda_availibility(self):

        try:
            if torch.cuda.is_available():
                show_info("Pytorch Using GPU")
            else:
                show_info("Pytorch Using CPU")
        except:
            print(traceback.format_exc())
            pass

    def check_gpufit_availibility(self):
        self.gpufit_available = False

        try:
            from pygpufit import gpufit as gf

            package_installed = True
        except:
            package_installed = False

        if package_installed:
            if not gf.cuda_available():
                show_info("Pygpufit not available due to missing CUDA")
            else:
                runtime_version, driver_version = gf.get_cuda_version()

                runtime_version = ".".join([str(v) for v in list(runtime_version)[:2]])
                driver_version = ".".join([str(v) for v in list(driver_version)][:2])

                runtime_version = float(runtime_version)
                driver_version = float(driver_version)

                self.gpufit_available = True

        else:
            show_info("Pygpufit not available due to missing package")

            import moltrack
            src_dir = moltrack.__file__.replace("\moltrack\__init__.py", "")
            show_info(f"Add pygpufit package to moltrack src directory [{src_dir}] to enable GPUFit.")

        if self.gpufit_available:
            if driver_version < runtime_version:
                show_info("Pygpufit may not work due to mismatched CUDA driver and runtime versions")
                show_info(f"GPUFit runtime: {runtime_version}, driver: {driver_version}")
            else:
                show_info("GPUFit available")
                show_info(f"GPUFit runtime: {runtime_version}, driver: {driver_version}")

            self.gui.smlm_fit_mode.addItem("GPUFit")
            self.gui.smlm_fit_mode.setCurrentIndex(1)
