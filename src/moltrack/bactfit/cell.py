import random
import string
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Event
from functools import partial
from scipy.spatial import distance
import os
from moltrack.bactfit.fit import BactFit
from shapely.geometry import Polygon, LineString

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
        self.cell_poles = None
        self.polynomial_params = None
        self.fit_error = None

        if cell_data is not None:

            for key in cell_data.keys():
                setattr(self, key, cell_data[key])

        if "name" not in cell_data.keys():
            #create random alphanumeric name
            self.name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    def optimise(self, refine_fit = True, fit_mode = "directed_hausdorff"):

        bf = BactFit()
        bf.fit_cell(self, refine_fit = refine_fit, fit_mode = fit_mode)




class CellList(object):
    def __init__(self, cell_list):
        self.data = cell_list

    def optimise(self, refine_fit=True, parallel=False, min_radius = -1, max_radius = -1,
            max_workers=None, progress_callback=None, fit_mode="directed_hausdorff",
            silence_tqdm=True):

        if len(self.data) > 0:

            bf = BactFit()

            cell_list = bf.fit_cell_list(self.data,
                refine_fit = refine_fit,
                parallel = parallel,
                fit_mode = fit_mode,
                min_radius = float(min_radius),
                max_radius = float(max_radius),
                max_workers = max_workers,
                progress_callback = progress_callback,
                silence_tqdm = silence_tqdm)

            self.data = cell_list

    def resize_polygon(self, polygon, n_points):

        outline = np.array(polygon.exterior.coords)
        outline = outline[1:]

        outline = LineString(outline)

        distances = np.linspace(0, outline.length, n_points)
        outline = LineString([outline.interpolate(distance) for distance in distances])

        outline = outline.coords

        polygon = Polygon(outline)

        return polygon

    def resize_line(self, line, n_points):

        distances = np.linspace(0, line.length, n_points)
        line = LineString([line.interpolate(distance) for distance in distances])

        return line


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

                        midline = self.resize_line(cell_midline, 6)
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









