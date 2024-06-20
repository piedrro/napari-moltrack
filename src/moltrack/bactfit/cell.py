import random
import string
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Manager, Event
from functools import partial
from scipy.spatial import distance
import os
import pickle
from shapely.geometry import Polygon, LineString, Point
from shapely.strtree import STRtree
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from picasso.render import render
import pyqtgraph as pg
import warnings

from moltrack.bactfit.fit import BactFit
from moltrack.bactfit.postprocess import cell_coordinate_transformation
from moltrack.bactfit.utils import resize_line, get_vertical


class ModelCell(object):

    def __init__(self, length = 10, width = 5, margin = 1):
        self.cell_polygon = None
        self.cell_midline = None
        self.cell_centerline = None
        self.width = width
        self.length = length
        self.margin = margin

        self.create_model_cell()

    def create_model_cell(self):

        x0 = y0 = self.width + self.margin

        # Define the coordinates of the line
        midline_x_coords = [x0, x0 + self.length]
        midline_y_coords = [y0, y0]
        midline_coords = list(zip(midline_x_coords, midline_y_coords))
        self.cell_midline = LineString(midline_coords)

        self.cell_polygon = self.cell_midline.buffer(self.width)

        y0 = self.width + self.margin
        x0 = self.margin
        centerline_x_coords = [x0, x0 + self.length + (self.width * 2)]
        centerline_y_coords = [y0, y0]
        centerline_coords = list(zip(centerline_x_coords, centerline_y_coords))
        centerline_coords = np.array(centerline_coords)

        self.cell_centerline = LineString(centerline_coords)

        self.cell_centerline = resize_line(self.cell_centerline, 100)


class Cell(object):

    def __init__(self, cell_data = None):

        self.cell_polygon = None
        self.cell_centre = None
        self.bbox = None
        self.height = None
        self.width = None
        self.vertical = None

        self.data = {}
        self.locs = []

        #fit data
        self.cell_fit = None
        self.cell_midline = None
        self.cell_centerline = None
        self.cell_poles = None
        self.cell_index = None
        self.polynomial_params = None
        self.fit_error = None

        if cell_data is not None:

            for key in cell_data.keys():
                setattr(self, key, cell_data[key])

        if "name" not in cell_data.keys():
            #create random alphanumeric name
            self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

        if hasattr(self, "midline_coords") and self.cell_midline is None:
            self.cell_midline = LineString(self.midline_coords)
            self.cell_poles = [self.cell_midline.coords[0], self.cell_midline.coords[-1]]

        if hasattr(self, "poly_params") and self.polynomial_params is None:
            self.polynomial_params = self.poly_params

        if self.cell_midline is not None and self.width is not None:
            self.cell_polygon = self.cell_midline.buffer(self.width)

        if self.vertical is None and self.cell_polygon is not None:
            self.vertical = get_vertical(self.cell_polygon)

    def __getstate__(self):
        # Return the state as a dictionary, omitting non-picklable attributes
        state = self.__dict__.copy()
        # Remove attributes that cannot be pickled
        # state.pop('non_picklable_attribute', None)
        return state

    def __setstate__(self, state):
        # Restore the state
        self.__dict__.update(state)

    def remove_locs_outside_polygon(self, locs=None):

        if locs is not None:
            self.locs = locs

        filtered_locs = []

        coords = np.stack([self.locs["x"], self.locs["y"]], axis=1)
        points = [Point(coord) for coord in coords]
        spatial_index = STRtree(points)

        possible_points = spatial_index.query(self.cell_polygon)

        polygon_point_indices = []

        for point_index in possible_points:
            point = points[point_index]

            if self.cell_polygon.contains(point):
                polygon_point_indices.append(point_index)

        if len(polygon_point_indices) > 0:
            polygon_locs = locs[polygon_point_indices]

            polygon_locs = pd.DataFrame(polygon_locs)

            if "cell_index" in polygon_locs.columns:
                polygon_locs = polygon_locs.drop(columns=["cell_index"])
            if "segmentation_index" in polygon_locs.columns:
                polygon_locs = polygon_locs.drop(columns=["segmentation_index"])

            polygon_locs = polygon_locs.to_records(index=False)

            filtered_locs.append(polygon_locs)

        if len(filtered_locs) > 0:
            filtered_locs = np.hstack(filtered_locs).view(np.recarray).copy()
            self.locs = filtered_locs
        else:
            self.locs = None

    def transform_locs(self, target_cell=None, locs=None, remove_outside_locs=True):

        if locs is not None:
            if remove_outside_locs:
                self.remove_locs_outside_polygon(locs)
            else:
                self.locs = locs

        if target_cell is not None and self.locs is not None:

            transformed_locs = cell_coordinate_transformation(self, target_cell)

            if len(transformed_locs) > 0:
                self.locs = transformed_locs
            else:
                self.locs = None


    def optimise(self, refine_fit = True, fit_mode = "directed_hausdorff",
            min_radius = -1, max_radius = -1):

        try:
            bf = BactFit(cell=self, refine_fit=refine_fit, fit_mode=fit_mode,
                min_radius=min_radius, max_radius=max_radius)

            self = bf.fit()

        except:
            print(traceback.format_exc())
            return None

        return self



class CellList(object):

    def __init__(self, cell_list):
        self.data = cell_list
        self.cell_names = []

        self.assign_cell_indices()
        self.assign_cell_names()

    def assign_cell_indices(self, reindex=False):

        if reindex == False:
            self.cell_indices = [cell.cell_index for cell in self.data
                                 if cell.cell_index is not None]
        else:
            self.cell_indices = []

        if len(self.cell_indices) == 0:

            for i, cell in enumerate(self.data):
                cell.cell_index = i
                self.cell_indices.append(i)

    def assign_cell_names(self):

        self.cell_names = [cell.name for cell in self.data if cell.name is not None]

        if len(self.cell_names) == 0:

            for cell in self.data:
                cell.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                self.cell_names.append(cell.name)

    def optimise(self, refine_fit=True, fit_mode="directed_hausdorff",
            min_radius = -1, max_radius = -1,
            max_workers=None, progress_callback=None, silence_tqdm=False, parallel=True, max_error=5):

        try:

            if len(self.data) == 0:
                return None

            if max_workers is None:
                max_workers = os.cpu_count()

            bf = BactFit(celllist=self,
                refine_fit=refine_fit, fit_mode=fit_mode,
                min_radius=min_radius, max_radius=max_radius,
                parallel=parallel, max_workers=max_workers,
                progress_callback=progress_callback,
                silence_tqdm=silence_tqdm)

            fitted_cells = bf.fit()

            self.data = [cell for cell in fitted_cells.data if cell.fit_error != None]
            self.data = [cell for cell in self.data if cell.fit_error < max_error]

            self.data = fitted_cells.data

        except:
            print(traceback.format_exc())
            return None

        return self

    def get_cell_fits(self, n_points = 100):

        fits = []
        poly_params = []
        cell_poles = []
        midlines = []
        cell_widths = []
        names = []

        for cell in self.data:
            if hasattr(cell, "cell_fit"):

                try:

                    cell_fit = cell.cell_fit
                    cell_width = cell.cell_width
                    cell_midline = cell.cell_midline
                    params = cell.polynomial_params
                    poles = cell.cell_poles

                    if cell_fit is not None:

                        name = cell.name
                        cell_fit = cell_fit.simplify(0.2)
                        seg = np.array(cell_fit.exterior.coords)

                        midline = resize_line(cell_midline, 6)
                        midline = np.array(midline.coords)

                        seg = seg[1:]

                        fits.append(seg)
                        names.append(name)
                        midlines.append(midline)
                        cell_widths.append(cell_width)
                        poly_params.append(params)
                        cell_poles.append(poles)

                except:
                    pass

        data = {"fits": fits,
                "midlines": midlines,
                "widths": cell_widths,
                "names": names,
                "poly_params": poly_params,
                "cell_poles": cell_poles,
                }

        return data

    def add_localisations(self, locs, remove_outside = True):

        if remove_outside:
            self.remove_locs_outside_polygons(locs)
        else:
            for cell in self.data:
                cell.locs = locs

    def remove_locs_outside_polygons(self, locs):

        polygon_list = [cell.cell_polygon for cell in self.data]

        coords = np.stack([locs["x"], locs["y"]], axis=1)
        points = [Point(coord) for coord in coords]
        spatial_index = STRtree(points)

        for polygon_index, polygon in enumerate(polygon_list):

            possible_points = spatial_index.query(polygon)

            polygon_point_indices = []

            for point_index in possible_points:
                point = points[point_index]

                if polygon.contains(point):
                    polygon_point_indices.append(point_index)

            if len(polygon_point_indices) > 0:
                polygon_locs = locs[polygon_point_indices]

                polygon_locs = pd.DataFrame(polygon_locs)

                if "cell_index" in polygon_locs.columns:
                    polygon_locs = polygon_locs.drop(columns=["cell_index"])
                if "segmentation_index" in polygon_locs.columns:
                    polygon_locs = polygon_locs.drop(columns=["segmentation_index"])

                polygon_locs["cell_index"] = polygon_index
                polygon_locs = polygon_locs.to_records(index=False)

                self.data[polygon_index].locs = polygon_locs

                # print(f"Cell {polygon_index} has {len(polygon_locs)} localisations")

    def transform_locs(self, target_cell=None, locs=None, remove_outside_locs=True, progress_callback=None):

        if locs is not None:
            if remove_outside_locs:
                self.remove_locs_outside_polygons(locs)
            else:
                for cell in self.data:
                    cell.locs = locs

        if target_cell is not None:

            compute_jobs = [list([cell,target_cell]) for cell in self.data if len(cell.locs) > 0]

            # compute_jobs = compute_jobs[:5]

            def compute_task(job):
                cell, target_cell = job
                cell = cell_coordinate_transformation(cell, target_cell)
                return cell

            n_jobs = len(compute_jobs)
            completed_jobs = 0

            # results = {}

            with ThreadPoolExecutor() as executor:

                futures = [executor.submit(compute_task, job) for job in compute_jobs]

                for future in as_completed(futures):
                    try:
                        result = future.result()

                        if result is not None:
                            cell = result
                            cell_index = cell.cell_index
                            self.data[cell_index].locs = cell.locs
                            # self.data[cell_index].locs = transformed_locs

                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()

                    completed_jobs += 1

                    if progress_callback is not None:
                        progress = 100 * (completed_jobs / n_jobs)
                        progress_callback.emit(progress)


    def get_locs(self):

        locs = []

        for cell in self.data:
            cell_locs = cell.locs
            if cell_locs is None:
                continue
            if len(cell_locs) == 0:
                continue

            locs.append(cell_locs)

        if len(locs) > 0:
            locs = np.hstack(locs).view(np.recarray).copy()
            return locs
        else:
            return None


    def plot_cell_heatmap(self, locs=None, color="red"):

        plot_locs = []

        for cell in self.data:
            locs = cell.locs
            if locs is not None and len(locs) > 0:
                plot_locs.append(locs)

        if len(plot_locs) > 0:
            locs = np.hstack(plot_locs).view(np.recarray).copy()

            # create heatmap
            heatmap, xedges, yedges = np.histogram2d(locs["x"], locs["y"], bins=30, density=False)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.rcParams["axes.grid"] = False
            plt.imshow(heatmap.T, extent=extent, origin='lower')
            plt.show()

    def plot_cell_render(self, oversampling=10, pixel_size=1, blur_method = "One-Pixel-Blur",
            locs=None, color="red"):

        plot_locs = []

        for cell in self.data:
            locs = cell.locs
            if locs is not None and len(locs) > 0:
                plot_locs.append(locs)

        if len(plot_locs) > 0:
            locs = np.hstack(plot_locs).view(np.recarray).copy()

            print(f"Rendering {len(locs)} localisations")

            xmin, xmax = int(np.min(locs["x"])), int(np.max(locs["x"]))
            ymin, ymax = int(np.min(locs["y"])), int(np.max(locs["y"]))

            h = ymax-ymin
            w = xmax-xmin

            viewport = [(float(0), float(0)), (float(h), float(w))]
            image_shape = (1, int(h), int(w))

            if blur_method == "One-Pixel-Blur":
                blur_method = "smooth"
            elif blur_method == "Global Localisation Precision":
                blur_method = "convolve"
            elif blur_method == "Individual Localisation Precision, iso":
                blur_method = "gaussian_iso"
            elif blur_method == "Individual Localisation Precision":
                blur_method = "gaussian"
            else:
                blur_method = None

            # n_rendered_locs, image = render(locs, viewport=viewport, blur_method=blur_method,
            #     min_blur_width=0, oversampling=oversampling, ang=0, )
            #
            # plt.imshow(image)
            # plt.show()




















