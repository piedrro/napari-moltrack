import concurrent.futures
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from multiprocessing import Manager

import mat4py
import numpy as np
import pandas as pd
import tifffile
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog
from shapely.geometry import Polygon

from moltrack.funcs.compute_utils import Worker
from moltrack.backend import image_source
from moltrack.model_views import refresh_row_projections


def import_image_data(dat, progress_dict=None, index=0):
    from moltrack.model_io import prepare_image
    callback = None if progress_dict is None else lambda progress: progress_dict.__setitem__(index, progress)
    return prepare_image(dat, progress_callback=callback)


def process_compute_jobs(compute_jobs, progress_callback=None, verbose=False):

    results = []

    if verbose:
        print(f"Processing {len(compute_jobs)} compute jobs.")

    cpu_count = int(multiprocessing.cpu_count() * 0.75)

    if len(compute_jobs) == 1:
        executor = ThreadPoolExecutor(max_workers=cpu_count)
    else:
        if len(compute_jobs) < cpu_count:
            cpu_count = len(compute_jobs)
        executor = ProcessPoolExecutor(max_workers=cpu_count)

    with Manager() as manager:
        progress_dict = manager.dict()

        with executor:
            # Submit all jobs and store the future objects
            futures = [executor.submit(import_image_data, job,
                progress_dict, i) for i, job in enumerate(compute_jobs)]

            while any(not future.done() for future in futures):
                # Calculate and emit progress
                total_progress = sum(progress_dict.values())
                overall_progress = int(total_progress / len(compute_jobs))
                if progress_callback is not None:
                    progress_callback.emit(overall_progress)
                time.sleep(0.1)  # Update frequency

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

            # Retrieve and process results
            results = [future.result() for future in futures if future.done()]

    if verbose:
        print("Finished processing compute jobs.")

    return results


def import_data(import_jobs, import_mode="data", progress_callback=None, verbose=False, publication=None, cancel_event=None):
    """Read captured import jobs without accessing a live widget."""
    from moltrack.model_processing import _cancel
    _cancel(cancel_event)
    jobs = process_compute_jobs(import_jobs, progress_callback, verbose)
    _cancel(cancel_event)
    return dict(jobs=jobs, import_mode=import_mode, publication=publication)


class _import_utils:

    def get_image_info(self, path):
        source = image_source(path)
        try:
            return len(source), source.shape, source.dtype, os.path.getsize(path)
        finally:
            source.close()

    def format_import_path(self, path):
        try:
            path = os.path.normpath(path)

            if os.name == "nt":
                if path.startswith("\\\\"):
                    path = "\\\\?\\UNC\\" + path[2:]

                    if "%" in str(path):
                        path = path.replace("%", "%%")

                if path.startswith("UNC"):
                    path = "\\\\?\\" + path

                    if "%" in str(path):
                        path = path.replace("%", "%%")

        except:
            print(traceback.format_exc())

        return path

    def populate_import_jobs(self, progress_callback=None, paths=[],
            import_crop_mode=None, import_limit=None, frame_averaging=None,
            multichannel_mode=None, channel_name=None, import_mode=None,
            pixel_size=None, exposure_time=None, **kwargs):

        import_jobs = []

        try:

            if import_crop_mode is None:
                import_crop_mode = self.gui.import_crop_mode.currentText()
            if import_limit is None:
                import_limit = self.gui.import_limit.currentText()
            if frame_averaging is None:
                frame_averaging = self.gui.frame_averaging.isChecked()
            if multichannel_mode is None:
                multichannel_mode = self.gui.import_multichannel_mode.currentText()
            if channel_name is None:
                channel_name = self.gui.import_channel_name.text()
            if import_mode is None:
                import_mode = self.gui.import_mode.currentText()
            if pixel_size is None:
                pixel_size = float(self.gui.import_pixel_size.value())
            if exposure_time is None:
                exposure_time = float(self.gui.import_exposure_time.value())

            if isinstance(paths, str):
                paths = [paths]

            for path_index, path in enumerate(paths):

                if multichannel_mode.lower() == "multi file":
                    dataset_name = self.gui.import_dataset_name.text()
                else:
                    path = self.format_import_path(path)
                    dataset_name = os.path.basename(path)

                image_dict = {"path": path,
                              "path_index": path_index,
                              "dataset_name": dataset_name,
                              "import_limit": import_limit,
                              "import_crop_mode": import_crop_mode,
                              "frame_averaging": frame_averaging,
                              "multichannel_mode": multichannel_mode,
                              "channel_name": channel_name,
                              "import_mode": import_mode,
                              "pixel_size": pixel_size,
                              "exposure_time": exposure_time, }

                import_jobs.append(image_dict)

        except:
            print(traceback.format_exc())

        return import_jobs


    def publish_import_images(self, import_list, *, clear=None, concatenate=None):
        from moltrack.model_io import import_images
        clear = self.gui.import_clear.isChecked() if clear is None else clear
        concatenate = self.gui.import_concatenate.isChecked() if concatenate is None else concatenate
        import_images(self.session, import_list, concatenate=concatenate, clear=clear)

    def publish_image_import(self, result):
        publication = result.get('publication')
        if publication is not None:
            from molcore import StaleResultError
            from molcore.staging import validate_expected
            event = getattr(self, 'stop_event', None)
            if (event is not None and event.is_set()) or self.session.model is not publication['owner']():
                show_info('Image import result was discarded after cancellation or project replacement')
                return
            try:
                validate_expected(self.session.model, publication.get('expected'))
            except StaleResultError:
                show_info('Image import result was discarded after project data changed')
                return
        if result['import_mode'].lower() == 'segmentation image':
            job = result['jobs'][0]
            from moltrack.model_io import import_images
            descriptor = dict(job, dataset_name='Segmentation Image', multichannel_mode='None', channel_name='Segmentation Image', role='segmentation_image')
            import_images(self.session, [descriptor])
            binding = 'Segmentation Image', 'Segmentation Image'
            self.session.set_annotations('segmentation_image_binding', binding)
            self.segmentation_image = self.session.image(*binding)[:]
            self._segmentation_source = self.segmentation_image
            self.segmentation_image_pixel_size = float(job['pixel_size'])
        else:
            options = {} if publication is None else {key: publication[key] for key in ('clear', 'concatenate')}
            self.publish_import_images(result['jobs'], **options)

    def import_data_finished(self):
        self.populate_dataset_selectors()
        self.update_active_image()
        self.draw_segmentation_image()
        self.update_fret_transform_target_channel()
        self.reset_slider()
        self.update_ui()

    def reset_slider(self):
        try:
            curent_step = self.viewer.dims.current_step
            curent_step = list(curent_step)
            curent_step[0] = 0
            curent_step = tuple(curent_step)
            self.viewer.dims.current_step = curent_step
        except:
            print(traceback.format_exc())

    def init_import_data(self, viewer=None, import_mode=None, import_path=None):

        try:

            if import_mode is None:
                import_mode = self.gui.import_mode.currentText()

            if import_path is None:
                desktop = os.path.expanduser("~/Desktop")

                if import_mode.lower() != ["segmentation image"]:
                    paths = QFileDialog.getOpenFileNames(self, "Open file", desktop, "Image files (*.tif *.fits)")[0]

                    paths = [path for path in paths if path != ""]

                else:
                    path = QFileDialog.getOpenFileName(self, "Open file", desktop, "Image files (*.tif *.fits)")[0]
                    paths = [path]
            else:
                if type(import_path) == str:
                    paths = [import_path]

            if paths != []:
                self.update_ui(init=True)

                import_jobs = self.populate_import_jobs(paths=paths)
                show_info(f"Importing {len(import_jobs)} images")
                from copy import deepcopy
                from weakref import ref
                from molcore.staging import state_token
                publication = dict(owner=ref(self.session.model),
                    expected=state_token(self.session.model),
                    clear=self.gui.import_clear.isChecked(),
                    concatenate=self.gui.import_concatenate.isChecked())
                self.worker = Worker(import_data, import_jobs=deepcopy(import_jobs), import_mode=import_mode,
                    verbose=self.verbose, publication=publication, cancel_event=getattr(self, 'stop_event', None))
                self.worker.signals.result.connect(self.publish_image_import)
                self.worker.signals.progress.connect(partial(self.moltrack_progress, progress_bar=self.gui.import_progressbar, ))
                self.worker.signals.finished.connect(self.import_data_finished)
                self.worker.signals.error.connect(self.update_ui)
                self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())


    def import_pos_out_locs(self, path, loc_cols=None):
        from molcore.files import read_table
        from moltrack.model_io import select_table_columns
        data = pd.DataFrame(read_table(path))
        data["frame"] = data["frame"] - 1
        return select_table_columns(data, loc_cols)

    def import_csv_locs(self, path, loc_cols):
        from molcore.files import read_table
        from moltrack.model_io import select_table_columns
        data = pd.DataFrame(read_table(path))
        return select_table_columns(data, loc_cols)

    def import_hdf5_locs(self, path, loc_cols):
        from molcore.interop import read_picasso
        from moltrack.model_io import select_table_columns
        columns, metadata = read_picasso(path)
        data = select_table_columns(pd.DataFrame(columns), loc_cols)
        data.attrs["picasso_info"] = metadata
        return data

    def import_localisations(self, import_dataset=None, import_channel=None,
            import_data=None, import_mode=None, path = None):

        try:

            if import_dataset is None:
                import_dataset = self.gui.locs_import_dataset.currentText()
            if import_channel is None:
                import_channel = self.gui.locs_import_channel.currentText()
            if import_data is None:
                import_data = self.gui.locs_import_data.currentText()
            if import_mode is None:
                import_mode = self.gui.locs_import_mode.currentText()

            if import_dataset == "" or import_channel == "":
                show_info("Localisation import requires image data to be loaded first")
                return None

            loc_cols = ["id", "localization_id", "source_localization_id", "track_id", "source_track_id", "dataset", "channel", "group", "particle", "frame",
                        "cell_index", "segmentation_index",
                        "x", "y", "photons", "bg", "sx", "sy", "lpx", "lpy",
                        "ellipticity", "net_gradient", "iterations",
                        "pixel_mean", "pixel_median", "pixel_sum", "pixel_min",
                        "pixel_max", "pixel_std", "pixel_mean_bg", "pixel_median_bg",
                        "pixel_sum_bg", "pixel_min_bg", "pixel_max_bg", "pixel_std_bg"]

            if "picasso" in import_mode.lower():
                ext = "HDF5 files (*.hdf5)"
                import_func = self.import_hdf5_locs
            elif import_mode == "CSV":
                ext = "CSV files (*.csv)"
                import_func = self.import_csv_locs
            elif import_mode == "POS.OUT":
                ext = "POS.OUT files (*.pos.out)"
                import_func = self.import_pos_out_locs
            else:
                return None

            if path is None:
                path = os.path.expanduser("~/Desktop")

                if hasattr(self, "session"):
                    if import_dataset in self.session.ordinary_datasets():
                        path = self.session.metadata(import_dataset)["path"]
                        if type(path) == list:
                            path = path[0]

                path = QFileDialog.getOpenFileName(self, "Open file", path, ext)[0]

            if path != "":

                locs = import_func(path, loc_cols)

                if locs is not None:
                    imported_metadata = locs.attrs.get("picasso_info")
                    locs["dataset"] = import_dataset
                    locs["channel"] = import_channel
                    locs = locs.to_records(index=False)
                    if imported_metadata is not None:
                        partitions = dict(self.session.annotations.get("partitions", {}))
                        dataset_annotations = dict(partitions.get(import_dataset, {}))
                        channel_annotations = dict(dataset_annotations.get(import_channel, {}))
                        channel_annotations["picasso_info"] = imported_metadata
                        dataset_annotations[import_channel] = channel_annotations
                        partitions[import_dataset] = dataset_annotations
                        self.session.set_annotations("partitions", partitions)


                    if import_data == "Localisations":

                        locs_cols = locs.dtype.names

                        if set(["dataset","channel","frame","x","y"]).issubset(locs_cols):

                            from moltrack.model_io import import_rows
                            import_rows(self.session, import_dataset, import_channel, locs)

                            imported_locs = self.get_locs(import_dataset, import_channel)
                            show_info(f"Imported {len(imported_locs)} localisations")

                            refresh_row_projections(self, localizations=True)

                            self.update_filter_criterion()
                            self.update_criterion_ranges()

                            self.update_SMLM_combos()

                        else:
                            show_info("Missing required columns for localisation data import")

                    if import_data == "Tracks":

                        locs_cols = locs.dtype.names
                        if set(["dataset","channel","particle", "frame","x","y"]).issubset(locs_cols):

                            from moltrack.model_io import import_rows
                            import_rows(self.session, import_dataset, import_channel, locs, tracks=True)
                            self.draw_tracks()

                            imported_tracks = self.get_tracks(import_dataset, import_channel)
                            show_info(f"Imported {len(imported_tracks)} tracks")

                            self.update_filter_criterion()
                            self.update_criterion_ranges()

                            self.update_track_filter_criterion()
                            self.update_track_criterion_ranges()

                            self.update_SMLM_combos()

                        else:
                            show_info("Missing required columns for tracking data import")
        except:
            print(traceback.format_exc())


    def import_binary_mask(self, path):

        try:

            mask = tifffile.imread(path)
            shapes = self.mask_to_shape(mask)

        except:
            shapes = None

        return shapes

    def import_cells(self, path):

        #import json file
        with open(path) as f:
            data = json.load(f)

        cell_names = data["name"]
        polygon_coords = data["polygon_coords"]
        midline_coords = data["midline_coords"]

        shapes = []
        shape_types = []
        properties = {"name": [], "cell": []}

        for cell_index, (name, polygon, midline) in enumerate(zip(
                cell_names, polygon_coords, midline_coords)):

            try:

                fit_params = {}
                if "name" in data.keys():
                    fit_params["name"] = data["name"][cell_index]
                if "width" in data.keys():
                    fit_params["width"] = data["width"][cell_index]
                if "poly_params" in data.keys():
                    fit_params["poly_params"] = data["poly_params"][cell_index]
                if "cell_poles" in data.keys():
                    fit_params["cell_poles"] = data["cell_poles"][cell_index]

                polygon = Polygon(polygon)
                polygon = np.array(polygon.exterior.coords)
                polygon = polygon[:-1]

                shapes.append(polygon)
                shape_types.append("polygon")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

                shapes.append(midline)
                shape_types.append("path")
                properties["name"].append(name)
                properties["cell"].append(fit_params)

            except:
                print(traceback.format_exc())

        shapes = {"shapes": shapes,
                  "shape_types": shape_types,
                  "properties": properties}

        return shapes

    def import_mesh(self, path):

        try:

            mat_data = mat4py.loadmat(path)

            mat_data = mat_data["cellList"]

            shapes = []

            for dat in mat_data:
                try:
                    if type(dat) == dict:
                        shape = np.array(dat["model"])
                        shape = np.fliplr(shape)
                        shape = shape[:-1]
                        shapes.append(shape)
                except:
                    pass

        except:
            shapes = None

        return shapes

    def import_shapes(self, import_data=None, import_mode=None, pixel_size=None, path=None):

        try:

            if import_data is None:
                import_data = self.gui.shapes_import_data.currentText()
            if import_mode is None:
                import_mode = self.gui.shapes_import_mode.currentText()
            if pixel_size is None:
                pixel_size = float(self.gui.shapes_import_pixel_size.value())

            self.segmentation_image_pixel_size = pixel_size

            if import_mode == "Binary Mask":
                import_func = self.import_binary_mask
                ext = "*.tif"
            elif import_mode == "JSON":
                import_func = self.import_cells
                ext = "*.json"
            elif import_mode == "Oufti/MicrobTracker Mesh":
                import_func = self.import_mesh
                ext = "*.mat"
            else:
                return None

            if path is None:
                path = os.path.expanduser("~/Desktop")
                path = QFileDialog.getOpenFileName(self, "Open file", path, ext)[0]

            if path != "":

                shape_data = import_func(path)

                if shape_data is not None:

                    if import_data == "Segmentations":

                        if import_mode in ["Binary Mask","Oufti/MicrobTracker Mesh"]:
                            self.initialise_segLayer(shape_data, pixel_size)

                        else:
                            shape_types = shape_data["shape_types"]
                            polygons = [shape_data["shapes"][i] for i, shape_type in enumerate(shape_types) if shape_type == "polygon"]

                            self.initialise_segLayer(polygons, pixel_size)

                    else:

                        shapes = shape_data["shapes"]
                        shape_types = shape_data["shape_types"]
                        properties = shape_data["properties"]

                        self.cellLayer = self.initialise_cellLayer(shapes=shapes,
                            shape_types=shape_types, properties=properties)

                        self.store_cell_shapes()

        except:
            print(traceback.format_exc())


    def toggle_moltrack_images_imported(self):

        self.moltrack_images_imported = True

    def import_moltrack_project(self, viewer=None, path=None):
        from copy import deepcopy
        from moltrack.model_io import load_project
        from moltrack.model_views import prepare_project_shape_layers

        candidate = None
        try:
            if path is None:
                path = QFileDialog.getOpenFileName(self, "Open file", os.path.expanduser("~/Desktop"), "*.moltrack")[0]
            if not path:
                return
            # Decode and validate the entire candidate before touching live state.
            candidate = load_project(path, include_images=self.gui.import_project_images.isChecked())
            prepared_layers = prepare_project_shape_layers(candidate)
        except Exception as error:
            if candidate is not None:
                candidate.close()
            show_info(f"Could not load MolTrack project: {error}")
            print(traceback.format_exc())
            return

        previous = self.session
        # Listeners capture a session owner. Fresh layers get fresh listeners.
        for attribute in ("segLayer", "cellLayer", "segmentation_layer", "image_layer", "loc_layer", "track_layer"):
            layer = getattr(self, attribute, None)
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
            if hasattr(self, attribute):
                delattr(self, attribute)
        for attribute in ("segmentation_image", "_segmentation_source", "segmentation_image_pixel_size", "stored_cells"):
            if hasattr(self, attribute):
                delattr(self, attribute)
        self.session = candidate
        self.active_dataset = None
        self.active_channel = None
        self.contrast_dict = deepcopy(candidate.annotations.get("contrast", {}))
        binding = candidate.annotations.get("segmentation_image_binding")
        if binding is not None and candidate.image_available(*binding):
            self.segmentation_image = candidate.image(*binding).as_dask()
            self._segmentation_source = self.segmentation_image
            self.segmentation_image_pixel_size = candidate.metadata(binding[0])["pixel_size"]
        self.import_data_finished()
        for name, method in (("segmentations", self.initialise_segLayer), ("cells", self.initialise_cellLayer)):
            annotation = candidate.annotations.get(name)
            if annotation is None:
                continue
            if name == "segmentations":
                method(shapes=annotation["data"], pixel_size=annotation["scale"][-1], restore_annotations=annotation, prepared_layer=prepared_layers[name])
            else:
                method(shapes=annotation["data"], shape_types=annotation["shape_types"], properties=annotation["properties"], restore_annotations=annotation, prepared_layer=prepared_layers[name])
        self.update_SMLM_combos()
        self.update_segmentation_combos()
        self.update_filter_criterion()
        self.update_criterion_ranges()
        self.update_track_filter_criterion()
        self.update_track_criterion_ranges()
        self.update_trackplot_options()
        self.plot_tracks(reset=True)
        self.update_ui()
        previous.close()
        for message in candidate.model.metadata.get("migration_report", []):
            print(message)
        show_info("MolTrack project loaded")
