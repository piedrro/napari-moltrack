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


def optimise_cell(cell, refine_fit=True, fit_mode="directed_hausdorff", min_radius=-1, max_radius=-1):
    cell.optimise(refine_fit=refine_fit, fit_mode=fit_mode, min_radius=min_radius, max_radius=max_radius)

    return cell

def resize_line(line, length):
    distances = np.linspace(0, line.length, length)
    line = LineString([line.interpolate(distance) for distance in distances])

    return line

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

def rotate_polygon(polygon, angle=90):
    origin = polygon.centroid.coords[0]
    polygon = shapely.affinity.rotate(polygon, angle=angle, origin=origin)

    return polygon

def rotate_linestring(linestring, angle=90):
    centroid = linestring.centroid
    origin = centroid.coords[0]

    linestring = shapely.affinity.rotate(linestring, angle=angle, origin=origin)

    return linestring

def get_vertical(polygon):

    minx, miny, maxx, maxy = polygon.bounds

    h = maxy - miny
    w = maxx - minx

    if h > w:
        vertical = True
    else:
        vertical = False

    return vertical


def fit_poly(coords, degree=2, constrained=True, constraining_points=[],
        minimise_curvature=True, curvature_weight = 0.1, degree_penalty=0.01, maxiter=50):

    warnings.filterwarnings("ignore", category=np.RankWarning)

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


def manual_fit(cell_coords, midline_coords, width = None):

    cell_polygon = Polygon(cell_coords)
    cell_outline = LineString(cell_coords)

    vertical = get_vertical(cell_polygon)

    n_medial_points = len(midline_coords)

    if vertical:
        cell_polygon = rotate_polygon(cell_polygon)
        cell_coords = np.array(cell_polygon.exterior.coords)
        cell_midline = LineString(midline_coords)
        cell_midline = rotate_linestring(cell_midline)
        midline_coords = np.array(cell_midline.coords)

    constraining_points = [midline_coords[0].tolist(),
                           midline_coords[-1].tolist()]

    medial_axis_fit, poly_params = fit_poly(midline_coords,
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
        cell_fit = rotate_polygon(cell_fit, angle=-90)
        cell_midline = rotate_linestring(cell_midline, angle=-90)

    cell_fit_coords = np.array(cell_fit.exterior.coords)
    cell_fit_coords = cell_fit_coords[:-1]

    cell_midline = resize_line(cell_midline, n_medial_points)
    midline_coords = np.array(cell_midline.coords)

    return cell_fit_coords, midline_coords, poly_params, cell_width

def resize_polygon(self, polygon, n_points):

    outline = np.array(polygon.exterior.coords)
    outline = outline[1:]

    outline = LineString(outline)

    distances = np.linspace(0, outline.length, n_points)
    outline = LineString([outline.interpolate(distance) for distance in distances])

    outline = outline.coords

    polygon = Polygon(outline)

    return polygon