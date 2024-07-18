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
import os

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
from moltrack.funcs.transform_utils import _transform_utils
from moltrack.funcs.management_utils import _management_utils
from moltrack.funcs.pixstats_utils import _pixstats_utils
from moltrack.funcs.trackplot_utils import _trackplot_utils

from moltrack.GUI.widget_ui import Ui_Frame as gui

subclasses = [_import_utils, _compute_utils,
              _events_utils, _segmentation_utils,
              _picasso_detect_utils, _loc_filter_utils,
              _picasso_render_utils, _tracking_utils,
              _export_utils, _segmentation_events,
              _bactfit_utils, _cell_events,
              oufti, _diffusion_utils, _cell_heatmap_utils,
              _track_filter_utils, _traces_utils,
              _transform_utils, _management_utils,
              _pixstats_utils, _trackplot_utils]

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
        self.update_shapes_import_options()

        self.draw_pixstats_mask(mode="locs")
        self.draw_pixstats_mask(mode="tracks")

        # create threadpool and stop event
        self.threadpool = QThreadPool()
        manager = Manager()
        self.stop_event = manager.Event()

        # self.import_dev_data()

    def import_dev_data(self):

        moltrack_root = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(os.path.dirname(moltrack_root))
        dev_data_path = os.path.join(package_dir, "dev_data")

        if os.path.exists(dev_data_path) is False:
            return

        image_path = os.path.join(dev_data_path, "image.fits")
        localisation_path = os.path.join(dev_data_path, "moltrack_localisations.csv")
        track_path = os.path.join(dev_data_path, "moltrack_tracks.csv")
        shapes_path = os.path.join(dev_data_path, "moltrack_shapes.json")

        self.init_import_data(
            import_mode="Data (Single Channel)",
            import_path=image_path)

        self.import_shapes(
            import_data="Segmentations",
            import_mode="JSON",
            path=shapes_path,
        )

        self.import_shapes(
            import_data="Cells",
            import_mode="JSON",
            path=shapes_path,
        )

        #wait for control to be enabled
        while self.gui.import_images.isEnabled() is False:
            QApplication.processEvents()

        self.import_localisations(
            import_dataset=self.active_dataset,
            import_channel=self.active_channel,
            import_mode="CSV",
            import_data="Localisations",
            path=localisation_path)

        self.import_localisations(
            import_dataset=self.active_dataset,
            import_channel=self.active_channel,
            import_mode="CSV",
            import_data="Tracks",
            path=track_path)

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

        self.gui.trackplot_graph_container.setLayout(QVBoxLayout())
        self.trackplot_canvas = CustomPyQTGraphWidget(self)
        self.gui.trackplot_graph_container.layout().addWidget(self.trackplot_canvas)

        self.heatmap_canvas = ImageView()
        self.gui.heatmap_graph_container.setLayout(QVBoxLayout())
        self.gui.heatmap_graph_container.layout().addWidget(self.heatmap_canvas)
        self.heatmap_canvas.ui.histogram.hide()
        self.heatmap_canvas.ui.roiBtn.hide()
        self.heatmap_canvas.ui.menuBtn.hide()

        self.locs_pixstats_canvas = ImageView()
        self.gui.locs_pixstats_graph_container.setLayout(QVBoxLayout())
        self.gui.locs_pixstats_graph_container.layout().addWidget(self.locs_pixstats_canvas)
        self.locs_pixstats_canvas.ui.histogram.hide()
        self.locs_pixstats_canvas.ui.roiBtn.hide()
        self.locs_pixstats_canvas.ui.menuBtn.hide()

        self.tracks_pixstats_canvas = ImageView()
        self.gui.tracks_pixstats_graph_container.setLayout(QVBoxLayout())
        self.gui.tracks_pixstats_graph_container.layout().addWidget(self.tracks_pixstats_canvas)
        self.tracks_pixstats_canvas.ui.histogram.hide()
        self.tracks_pixstats_canvas.ui.roiBtn.hide()
        self.tracks_pixstats_canvas.ui.menuBtn.hide()

        self.dataset_dict = {}
        self.localisation_dict = {}
        self.tracking_dict = {}
        self.contrast_dict = {}
        self.diffusion_dict = {}

        self.active_dataset = None
        self.active_channel = None

        self.transform_matrix = None

        self.verbose = False

        self.segmentation_mode = "panzoom"
        self.interface_mode = "segment"

        self.trackplot_tracks = None

        self.moltrack_metrics = {"Mean Squared Displacement": "msd",
                                 "Speed": "speed",
                                 "Apparent Diffusion Coefficient": "D*",
                                 "X": "x",
                                 "Y": "y",
                                 "Photons": "photons",
                                 "Background": "bg",
                                 "PSF width X": "sx",
                                 "PSF width Y": "sy",
                                 "Localisation Precision X": "lpx",
                                 "Localisation Precision Y": "lpy",
                                 "Ellipticity": "ellipticity",
                                 "Pixel Mean": "pixel_mean",
                                 "Pixel StdDev": "pixel_std",
                                 "Pixel Median": "pixel_median",
                                 "Pixel Min": "pixel_min",
                                 "Pixel Max": "pixel_max",
                                 "Pixel Sum": "pixel_sum",
                                 "Pixel Mean FRET": "pixel_mean_fret",
                                 "Pixel StdDev FRET": "pixel_std_fret",
                                 "Pixel Median FRET": "pixel_median_fret",
                                 "Pixel Min FRET": "pixel_min_fret",
                                 "Pixel Max FRET": "pixel_max_fret",
                                 "Pixel Sum FRET": "pixel_sum_fret",
                                 }

        # self.gui.tracks_pixstats_fret.hide()
        # self.gui.locs_pixstats_fret.hide()

        np.seterr(all='ignore')


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
        self.gui.picasso_filter_channel.currentIndexChanged.connect(self.update_filter_criterion)
        self.gui.filter_criterion.currentIndexChanged.connect(self.update_criterion_ranges)
        self.gui.filter_subtract_bg.stateChanged.connect(self.update_criterion_ranges)
        self.gui.filter_localisations.clicked.connect(self.pixseq_filter_localisations)

        self.gui.tracks_pixstats_spot_size.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="tracks"))
        self.gui.tracks_pixstats_spot_shape.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="tracks"))
        self.gui.tracks_pixstats_bg_width.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="tracks"))
        self.gui.tracks_pixstats_bg_buffer.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="tracks"))

        self.gui.locs_pixstats_spot_size.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="locs"))
        self.gui.locs_pixstats_spot_shape.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="locs"))
        self.gui.locs_pixstats_bg_width.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="locs"))
        self.gui.locs_pixstats_bg_buffer.currentIndexChanged.connect(partial(self.draw_pixstats_mask, mode="locs"))

        self.gui.track_filter_dataset.currentIndexChanged.connect(self.update_track_filter_criterion)
        self.gui.track_filter_channel.currentIndexChanged.connect(self.update_track_filter_criterion)
        self.gui.track_filter_criterion.currentIndexChanged.connect(self.update_track_filter_metric)
        self.gui.track_filter_metric.currentIndexChanged.connect(self.update_track_criterion_ranges)
        self.gui.track_filter_subtract_bg.stateChanged.connect(self.update_track_criterion_ranges)
        self.gui.filter_tracks.clicked.connect(self.filter_tracks)
        self.gui.compute_track_stats.clicked.connect(self.initialise_track_stats)

        self.gui.locs_pixstats_compute.clicked.connect(partial(self.initialise_pixstats, mode="locs"))
        self.gui.tracks_pixstats_compute.clicked.connect(partial(self.initialise_pixstats, mode="tracks"))

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
        self.gui.export_heatmap_locs.clicked.connect(self.export_heatmap_locs)
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

        self.gui.import_shapes.clicked.connect(self.import_shapes)
        self.gui.shapes_import_data.currentIndexChanged.connect(self.update_shapes_import_options)

        self.gui.import_tform.clicked.connect(self.import_fret_transform_matrix)
        self.gui.compute_tform.clicked.connect(self.compute_fret_transform_matrix)
        self.gui.apply_tform.clicked.connect(self.apply_fret_transform_matrix)
        self.gui.tform_compute_channel.currentTextChanged.connect(self.update_fret_transform_target_channel)

        self.gui.copy_locs.clicked.connect(partial(self.copy_data, mode="locs"))
        self.gui.copy_tracks.clicked.connect(partial(self.copy_data, mode="tracks"))

        self.gui.delete_locs.clicked.connect(partial(self.delete_data, mode="locs"))
        self.gui.delete_tracks.clicked.connect(partial(self.delete_data, mode="tracks"))

        self.gui.trackplot_dataset.currentIndexChanged.connect(self.update_trackplot_options)
        self.gui.trackplot_channel.currentIndexChanged.connect(self.update_trackplot_options)
        self.gui.trackplot_slider.valueChanged.connect(self.update_trackplot_slider)
        self.gui.trackplot_metric1.currentIndexChanged.connect(partial(self.plot_tracks, reset=True))
        self.gui.trackplot_metric2.currentIndexChanged.connect(partial(self.plot_tracks, reset=True))
        self.gui.trackplot_metric3.currentIndexChanged.connect(partial(self.plot_tracks, reset=True))
        self.gui.trackplot_metric4.currentIndexChanged.connect(partial(self.plot_tracks, reset=True))
        self.gui.trackplot_focus.stateChanged.connect(self.plot_tracks)
        self.gui.trackplot_highlight.stateChanged.connect(self.plot_tracks)
        self.gui.trackplot_slider.valueChanged.connect(self.plot_tracks)
        self.gui.trackplot_subtrack_background.stateChanged.connect(self.plot_tracks)

        self.gui.trackplot_highlight.stateChanged.connect(self.reset_tracks)

    def devfunc(self, viewer=None):

        # self.update_render_length_range()
        # self.update_render_msd_range()

        self.update_ui()
        # self.update_filter_criterion()
        # self.update_criterion_ranges()

        self.get_trackplot_metrics()

        # self.draw_localisations()
        # self.export_celllist()

        # self.update_traces_export_options()

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
        # self.gui.track_filter_channel.currentIndexChanged.connect(self.update_track_filter_criterion)



    def export_celllist(self):

        try:

            from moltrack.bactfit.fileIO import load, save

            celllist = self.populate_celllist()

            cell = celllist.data[0]

            save("celllist.h5", cell)
            load("celllist.h5")

        except:
            print(traceback.format_exc())
            pass

    def initialise_keybindings(self):

        self.viewer.bind_key("F1", self.devfunc, overwrite=True,)

        self.viewer.bind_key("Control-Right", func=lambda event: self.moltract_translation(direction="right"), overwrite=True, )
        self.viewer.bind_key("Control-Left", func=lambda event: self.moltract_translation(direction="left"), overwrite=True, )
        self.viewer.bind_key("Control-Up", func=lambda event: self.moltract_translation(direction="up"), overwrite=True, )
        self.viewer.bind_key("Control-Down", func=lambda event: self.moltract_translation(direction="down"), overwrite=True, )
        self.viewer.bind_key("Control-Z", func=self.moltrack_undo, overwrite=True, )

        self.register_shape_layer_keybinds(self.viewer)

    def remove_keybindings(self):

        self.viewer.bind_key("F1", None)

        self.viewer.bind_key("Control-Right", func=None)
        self.viewer.bind_key("Control-Left", func=None)
        self.viewer.bind_key("Control-Up", func=None)
        self.viewer.bind_key("Control-Down", func=None)
        self.viewer.bind_key("Control-Z", func=None)

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
