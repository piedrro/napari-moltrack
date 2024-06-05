import numpy as np
import traceback
from moltrack.funcs.compute_utils import Worker
from moltrack.bactfit.fit import BactFit
from moltrack.bactfit.preprocess import data_to_cells
from moltrack.bactfit.fit import BactFit
from functools import partial
from shapely.geometry import Polygon, LineString

class _bactfit_utils:


    def update_cells(self, event):

        try:

            if event.action == "changed":

                selected_index = list(self.cellLayer.selected_data)

                if len(selected_index) == 1:

                    selected_index = selected_index[0]

                    name_list = self.cellLayer.properties["name"].copy()
                    shape_types = self.cellLayer.shape_type.copy()
                    shapes = self.cellLayer.data.copy()

                    name = name_list[selected_index]
                    selected_shape_type = shape_types[selected_index]

                    if selected_shape_type == "path":

                        selected_cells = [i for i, n in enumerate(name_list) if n == name]

                        if len(selected_cells) == 2:

                            path_index = [i for i in selected_cells if shape_types[i] == "path"]
                            polygon_index = [i for i in selected_cells if shape_types[i] == "polygon"]

                            if len(path_index) == 1 and len(polygon_index) == 1:

                                polygon = shapes[polygon_index[0]]
                                midline = shapes[path_index[0]]

                                bf = BactFit()

                                cell_fit = bf.manual_fit(polygon, midline)

                                if cell_fit is not None:

                                    self.cellLayer.events.disconnect(self.update_cells)

                                    shapes = self.cellLayer.data.copy()
                                    shapes[polygon_index[0]] = cell_fit
                                    self.cellLayer.data = shapes
                                    self.cellLayer.refresh()

                                    self.cellLayer.events.data.connect(self.update_cells)

                    if selected_shape_type == "polygon":

                        selected_cells = [i for i, n in enumerate(name_list) if n == name]

                        if len(selected_cells) == 2:

                            path_index = [i for i in selected_cells if shape_types[i] == "path"]
                            polygon_index = [i for i in selected_cells if shape_types[i] == "polygon"]

                            if len(path_index) == 1 and len(polygon_index) == 1:

                                polygon_coords = shapes[polygon_index[0]]
                                midline_coords = shapes[path_index[0]]

                                cell_polygon = Polygon(polygon_coords)
                                cell_midline = LineString(midline_coords)

                                polygon_origin = cell_polygon.centroid.coords[0]
                                midline_origin = cell_midline.centroid.coords[0]

                                x_shift = polygon_origin[0] - midline_origin[0]
                                y_shift = polygon_origin[1] - midline_origin[1]

                                midline_coords[:, 0] += x_shift
                                midline_coords[:, 1] += y_shift

                                self.cellLayer.events.disconnect(self.update_cells)

                                shapes = self.cellLayer.data.copy()
                                shapes[path_index[0]] = midline_coords

                                self.cellLayer.data = shapes
                                self.cellLayer.refresh()

                                self.cellLayer.events.data.connect(self.update_cells)



        except:
            pass

    def initialise_cellLayer(self, shapes = None, shape_types = None, properties = None):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Cells" in layer_names:
            self.viewer.layers.remove("Cells")

        if hasattr(self, "cellLayer"):
            del self.cellLayer

        if shapes is not None:
            self.cellLayer = self.viewer.add_shapes(shapes, properties=properties,
                shape_type=shape_types, name="Cells",)
        else:
            self.cellLayer = self.viewer.add_shapes(name="Cells", shape_type="polygon",
                opacity=0.5, face_color="red", edge_color="black", edge_width=1)

        self.cellLayer.events.disconnect(self.update_cells)
        self.cellLayer.events.data.connect(self.update_cells)

        return self.cellLayer

    def run_bactfit_finished(self):
        self.update_ui()

    def run_bactfit_results(self, cell_list):

        if cell_list is None:
            return

        data = cell_list.get_cell_fits()

        cell_names = data["names"]
        cell_fits = data["fits"]
        cell_midlines = data["midlines"]

        layer_names = [layer.name for layer in self.viewer.layers]

        if "fitted_cells" in layer_names:
            self.viewer.layers.remove("fitted_cells")

        properties = {"name": cell_names*2,}

        shapes = cell_midlines + cell_fits
        shape_types = ["path"] * len(cell_midlines) + ["polygon"] * len(cell_fits)

        self.cellLayer = self.initialise_cellLayer(shapes=shapes,
            shape_types=shape_types, properties=properties)

    def run_bactfit(self, segmentations, progress_callback=None):

        try:

            min_radius = float(self.gui.fit_min_radius.value())
            max_radius = float(self.gui.fit_max_radius.value())

            cell_list = data_to_cells(segmentations)

            cell_list.optimise(refine_fit=True, parallel=True,
                min_radius=min_radius, max_radius=max_radius,
                progress_callback=progress_callback)

        except:
            print(traceback.format_exc())
            return None

        return cell_list

    def initialise_bactfit(self):

        if hasattr(self, "segLayer"):

            segmentations = self.segLayer.data

            if len(segmentations) == 0:
                return

            self.update_ui(init=True)

            worker = Worker(self.run_bactfit, segmentations)
            worker.signals.progress.connect(partial(self.moltrack_progress,
                    progress_bar=self.gui.bactfit_progressbar))
            worker.signals.result.connect(self.run_bactfit_results)
            worker.signals.finished.connect(self.run_bactfit_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.threadpool.start(worker)






