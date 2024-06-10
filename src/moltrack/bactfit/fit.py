import tifffile 
import numpy as np
import cv2
import os
import traceback
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis, binary_opening, disk
from scipy.interpolate import CubicSpline, BSpline
from skimage.draw import polygon
from shapely.geometry import LineString, Point, LinearRing, Polygon
from shapely.affinity import rotate
import matplotlib.path as mpltPath
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
from scipy.interpolate import interp1d
import math
from scipy.interpolate import splrep, splev
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import shapely
from scipy.spatial.distance import directed_hausdorff
import shapely
from scipy.spatial import distance
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Event
from functools import partial
import warnings

class BactFit(object):

    @staticmethod
    def get_vertical(polygon):

        minx, miny, maxx, maxy = polygon.bounds

        h = maxy - miny
        w = maxx - minx

        if h > w:
            vertical = True
        else:
            vertical = False

        return vertical

    def manual_fit(self, cell_coords, midline_coords, width = None):

        cell_polygon = Polygon(cell_coords)
        cell_outline = LineString(cell_coords)

        vertical = BactFit.get_vertical(cell_polygon)

        n_medial_points = len(midline_coords)

        if vertical:
            cell_polygon = BactFit.rotate_polygon(cell_polygon)
            cell_coords = np.array(cell_polygon.exterior.coords)
            cell_midline = LineString(midline_coords)
            cell_midline = BactFit.rotate_linestring(cell_midline)
            midline_coords = np.array(cell_midline.coords)

        constraining_points = [midline_coords[0].tolist(),
                               midline_coords[-1].tolist()]

        medial_axis_fit, poly_params = BactFit.fit_poly(midline_coords,
            degree=[1, 2, 3], maxiter=100, minimise_curvature=False,
            constraining_points=constraining_points, constrained=False)

        if width is None:
            centroid = cell_polygon.centroid
            cell_width = cell_outline.distance(centroid)
        else:
            cell_width = width

        cell_midline = LineString(medial_axis_fit)
        cell_fit = cell_midline.buffer(cell_width)

        if vertical:
            cell_fit = BactFit.rotate_polygon(cell_fit, angle=-90)
            cell_midline = BactFit.rotate_linestring(cell_midline, angle=-90)

        cell_fit_coords = np.array(cell_fit.exterior.coords)
        cell_fit_coords = cell_fit_coords[:-1]

        cell_midline = BactFit.resize_line(cell_midline, n_medial_points)
        midline_coords = np.array(cell_midline.coords)

        return cell_fit_coords, midline_coords, poly_params, cell_width



    @staticmethod
    def get_polygon_medial_axis(outline, refine=True):

        if type(outline) == LineString:
            polygon_outline = outline
            polygon = Polygon(outline.coords)
        elif type(outline) == Polygon:
            polygon = outline
            polygon_outline = LineString(polygon.exterior.coords)
        else:
            return None, None

        if len(polygon_outline.coords) < 200:
            polygon_outline = BactFit.resize_line(polygon_outline, 200)
            polygon = Polygon(polygon_outline.coords)

        # Extract the exterior coordinates of the polygon
        exterior_coords = np.array(polygon.exterior.coords)

        exterior_coords = BactFit.moving_average(exterior_coords, padding=10, iterations=2)

        # Compute the Voronoi diagram of the exterior coordinates
        vor = Voronoi(exterior_coords)

        # Function to check if a point is inside the polygon
        def point_in_polygon(point, polygon):
            return polygon.contains(Point(point))

        # Extract the medial axis points from the Voronoi vertices

        coords = []

        for i, region in enumerate(vor.regions):
            if -1 not in region:
                try:
                    coords.append(vor.vertices[i].tolist())
                except:
                    pass

        coords = [point for point in coords if point_in_polygon(point, polygon)]

        centroid = polygon.centroid
        cell_radius = polygon_outline.distance(centroid)

        if refine:
            coords = [p for p in coords if polygon_outline.distance(Point(p)) > cell_radius * 0.8]
            coords = np.array(coords)

        return np.array(coords), cell_radius

    @staticmethod
    def resize_line(line, length):
        distances = np.linspace(0, line.length, length)
        line = LineString([line.interpolate(distance) for distance in distances])

        return line

    @staticmethod
    def moving_average(line, padding=5, iterations=1):
        x, y = line[:, 0], line[:, 1]

        x = np.concatenate((x[-padding:], x, x[:padding]))
        y = np.concatenate((y[-padding:], y, y[:padding]))

        for i in range(iterations):
            y = np.convolve(y, np.ones(padding), 'same') / padding
            x = np.convolve(x, np.ones(padding), 'same') / padding

            x = np.array(x)
            y = np.array(y)

        x = x[padding:-padding]
        y = y[padding:-padding]

        line = np.stack([x, y]).T

        return line

    @staticmethod
    def bactfit_result(cell, params, cell_polygon, poly_params,
            fit_mode = "directed_hausdorff"):

        x_min = params[0]
        x_max = params[1]
        cell_width = params[2]
        x_offset = params[3]
        y_offset = params[4]
        angle = params[5]

        p = np.poly1d(poly_params)
        x_fitted = np.linspace(x_min, x_max, num=20)
        y_fitted = p(x_fitted)

        x_fitted += x_offset
        y_fitted += y_offset

        midline_coords = np.column_stack((x_fitted, y_fitted))
        midline = LineString(midline_coords)

        midline = BactFit.rotate_linestring(midline, angle=angle)

        midline_coords = np.array(midline.coords)
        cell_poles = [midline_coords[0], midline_coords[-1]]

        cell_fit = midline.buffer(cell_width)

        distance = BactFit.compute_bacfit_distance(cell_fit, cell_polygon, fit_mode)

        cell.cell_midline = midline
        cell.cell_fit = cell_fit
        cell.cell_poles = cell_poles
        cell.polynomial_params = poly_params
        cell.fit_error = distance
        cell.cell_width = cell_width

        return cell


    @staticmethod
    def refine_function(params, cell_polygon, poly_params,
            fit_mode="directed_hausdorff"):

        """
        Objective function to minimize: the Hausdorff distance between the buffered spline and the target contour.
        """

        try:
            params = list(params)

            x_min = params[0]
            x_max = params[1]
            cell_width = params[2]
            x_offset = params[3]
            y_offset = params[4]
            angle = params[5]

            p = np.poly1d(poly_params)
            x_fitted = np.linspace(x_min, x_max, num=10)
            y_fitted = p(x_fitted)

            x_fitted += x_offset
            y_fitted += y_offset

            midline_coords = np.column_stack((x_fitted, y_fitted))

            midline = LineString(midline_coords)
            midline = BactFit.rotate_linestring(midline, angle=angle)

            midline_buffer = midline.buffer(cell_width)

            distance = BactFit.compute_bacfit_distance(midline_buffer,
                cell_polygon, fit_mode)

        except:
            distance = np.inf

        return distance

    @staticmethod
    def compute_bacfit_distance(midline_buffer, cell_polygon,
            fit_mode = "directed_hausdorff"):

        try:
            if fit_mode == "hausdorff":
                # Calculate the Hausdorff distance between the buffered spline and the target contour
                distance = midline_buffer.hausdorff_distance(cell_polygon)
            elif fit_mode == "directed_hausdorff":
                # Calculate directed Hausdorff distance in both directions
                buffer_points = np.array(midline_buffer.exterior.coords)
                contour_points = np.array(cell_polygon.exterior.coords)
                dist1 = directed_hausdorff(buffer_points, contour_points)[0]
                dist2 = directed_hausdorff(contour_points, buffer_points)[0]
                distance = dist1 + dist2

        except:
            print(traceback.format_exc())
            distance = np.inf

        return distance

    @staticmethod
    def get_poly_coords(x, y, coefficients, margin=0, n_points=10):
        x1 = np.min(x) - margin
        x2 = np.max(x) + margin

        p = np.poly1d(coefficients)
        x_fitted = np.linspace(x1, x2, num=n_points)
        y_fitted = p(x_fitted)

        return np.column_stack((x_fitted, y_fitted))

    @staticmethod
    def fit_poly(coords, degree=2, constrained=True, constraining_points=[],
            minimise_curvature=True, curvature_weight = 0.1, degree_penalty=0.01, maxiter=50):
        def polynomial_fit(params, x):
            # Reverse the parameters to match np.polyfit order
            params = params[::-1]
            return sum(p * x ** i for i, p in enumerate(params))

        def objective_function(params, x, y, minimise_curvature=True):
            fit_error = np.sum((polynomial_fit(params, x) - y) ** 2)

            if minimise_curvature:
                curvature_penalty = np.sum(np.diff(params, n=2) ** 2)
                fit_error = fit_error + (curvature_penalty*curvature_weight)

            fit_error = fit_error + (degree_penalty * len(params))

            return fit_error

        def constraint_function(params, x_val, y_val):
            return polynomial_fit(params, x_val) - y_val

        def get_coords(x, y, coefficients, margin=0, n_points=10):
            x1 = np.min(x) - margin
            x2 = np.max(x) + margin

            p = np.poly1d(coefficients)
            x_fitted = np.linspace(x1, x2, num=n_points)
            y_fitted = p(x_fitted)

            return np.column_stack((x_fitted, y_fitted))

        x = coords[:, 0]
        y = coords[:, 1]
        constraints = []

        param_list = []
        error_list = []
        success_list = []

        if constrained and len(constraining_points) > 0:
            for point in constraining_points:
                if len(point) == 2:
                    constraints.append({'type': 'eq', 'fun': constraint_function, 'args': point})

        if type(degree) != list:
            degree = [degree]

        for deg in degree:
            params = np.polyfit(x, y, deg)

            result = minimize(objective_function, params, args=(x, y, minimise_curvature),
                constraints=constraints, tol=1e-6, options={'maxiter': maxiter})

            param_list.append(result.x)
            error_list.append(result.fun)
            success_list.append(result.success)

        min_error_index = error_list.index(min(error_list))

        best_params = param_list[min_error_index]

        fitted_poly = get_coords(x, y, best_params)

        return fitted_poly, list(best_params)

    @staticmethod
    def register_fit_data(cell, cell_centre=[], vertical=False):

        if vertical:

            cell_fit = cell.cell_fit
            cell_midline = cell.cell_midline

            cell_fit = BactFit.rotate_polygon(cell_fit, angle=-90)
            cell_midline = BactFit.rotate_linestring(cell_midline, angle=-90)

            midline_coords = np.array(cell_midline.coords)
            cell_poles = [midline_coords[0], midline_coords[-1]]

            cell.cell_fit = cell_fit
            cell.cell_midline = cell_midline
            cell.cell_poles = cell_poles

        return cell

    @staticmethod
    def rotate_polygon(polygon, angle=90):
        origin = polygon.centroid.coords[0]
        polygon = shapely.affinity.rotate(polygon, angle=angle, origin=origin)

        return polygon

    @staticmethod
    def rotate_linestring(linestring, angle=90):
        centroid = linestring.centroid
        origin = centroid.coords[0]

        linestring = shapely.affinity.rotate(linestring, angle=angle, origin=origin)

        return linestring

    @staticmethod
    def fit_cell(cell, refine_fit = True, min_radius = -1, max_radius = -1,
            fit_mode = "directed_hausdorff", **kwargs):


        try:

            cell_polygon = cell.cell_polygon
            vertical = cell.vertical

            if vertical:
                cell_polygon = BactFit.rotate_polygon(cell_polygon)

            medial_axis_coords, radius = BactFit.get_polygon_medial_axis(cell_polygon)

            if min_radius > 0:
                radius = max(radius, min_radius)
            if max_radius > 0:
                radius = min(radius, max_radius)

            medial_axis_fit, poly_params = BactFit.fit_poly(medial_axis_coords,
                degree=[1, 2, 3], maxiter=100, minimise_curvature=False)

            x_min = np.min(medial_axis_fit[:, 0])
            x_max = np.max(medial_axis_fit[:, 0])
            x_offset = 0
            y_offset = 0
            rotation = 0
            params = [x_min, x_max, radius, x_offset, y_offset, rotation]

            if refine_fit:

                warnings.filterwarnings("ignore", category=RuntimeWarning)

                bounds = [(None, None),  # x_min
                          (None, None),  # x_max
                          (None, None),  # radius
                          (None, None),  # x_offset
                          (None, None),  # y_offset
                          (None, None)]  # rotation

                if min_radius > 0:
                    bounds[2] = (min_radius, None)
                if max_radius > 0:
                    bounds[2] = (None, max_radius)

                result = minimize(BactFit.refine_function, params,
                    args=(cell_polygon, poly_params, fit_mode),
                    tol=1e-6, options={'maxiter': 500}, bounds=bounds)

                params = result.x

            cell = BactFit.bactfit_result(cell, params, cell_polygon,
                poly_params, fit_mode)
            cell = BactFit.register_fit_data(cell, vertical=vertical)

        except:
            print(traceback.format_exc())
            pass

        return cell


    @staticmethod
    def fit_cell_list(cell_list, refine_fit = True,
            fit_mode = "directed_hausdorff",
            min_radius = -1, max_radius = -1,
            parallel = False, max_workers = None,
            progress_callback = None, silence_tqdm = False, **kwargs):

        num_cells = len(cell_list)

        if parallel:

            if max_workers == None:
                max_workers = os.cpu_count()

            with ProcessPoolExecutor(max_workers=max_workers) as executor:

                futures = {executor.submit(BactFit.fit_cell,
                    cell_obj,
                    refine_fit=refine_fit,
                    fit_mode=fit_mode,
                    min_radius=min_radius,
                    max_radius=max_radius): cell_obj for cell_obj in cell_list}

                completed = 0
                for future in as_completed(futures):
                    cell = future.result()
                    cell_obj = futures[future]
                    idx = cell_list.index(cell_obj)
                    cell_list[idx] = cell

                    completed += 1
                    if progress_callback is not None:
                        progress = (completed / num_cells) * 100
                        progress_callback.emit(progress)

        else:

            iter = 0

            for cell_index, cell in enumerate(cell_list):

                cell = BactFit.fit_cell(cell, refine_fit=refine_fit,
                    fit_mode=fit_mode, min_radius=min_radius,
                    max_radius=max_radius)

                cell_list[cell_index] = cell

                if progress_callback is not None:
                    progress = ((iter + 1) / num_cells)*100
                    progress_callback.emit(progress)

                iter += 1

        return cell_list
