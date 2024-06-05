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
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Event
from functools import partial


class BactFit:
    def __init__(self, verbose = True):

        print("bactfit initialised")
        
        self.contours = []
        self.cell_masks = []
        self.verbose = verbose


    def fit_cell_masks(self, masks = [], fit = True, parallel = False):

        if type(masks) != list:
            masks = [masks]

        self.populate_cell_dataset(masks, edge_cells=False)

        self.run_mask_fit(fit = fit,
            parallel = parallel)


    def fit_cell_contours(self, contours = [],
            fit = True, parallel = False, progress_callback = None):

        if type(contours) != list:
            contours = [contours]

        contour_dataset = self.populate_contour_dataset(contours)

        if len(contour_dataset) == 0:
            return None

        fit_data = self.run_contour_fit(contour_dataset, fit=fit,
            parallel=parallel, progress_callback=progress_callback)

        return fit_data


    def populate_contour_dataset(self, contours):

        contour_dataset = []

        for contour_index, contour in enumerate(contours):

            cell_polygon = Polygon(contour)

            centroid = cell_polygon.centroid

            cell_centre = [centroid.x, centroid.y]

            minx, miny, maxx, maxy = cell_polygon.bounds

            h = maxy - miny
            w = maxx - minx

            if h > w:
                vertical = True
            else:
                vertical = False

            print(f"Contour {contour_index} h:{h} w:{w}, vertical:{vertical}")

            dat = {"contour_index":contour_index,
                   "cell_polygon":cell_polygon,
                   "cell_center":cell_centre,
                   "vertical":vertical,
                   }

            contour_dataset.append(dat)

        if self.verbose:
            print(f"Imported {len(contour_dataset)} contour(s)")

        return contour_dataset









    def check_edge_cell(self, cnt, mask, buffer=5):

        edge = False

        try:

            cell_mask_bbox = cv2.boundingRect(cnt)
            [x, y, w, h] = cell_mask_bbox
            [x1, y1, x2, y2] = [x, y, x + w, y + h]
            bx1, by1, bx2, by2 = [x1 - buffer, y1 - buffer, x2 + buffer, y2 + buffer]

            if bx1 < 0:
                edge = True
            if by1 < 0:
                edge = True
            if bx2 > mask.shape[1]:
                edge = True
            if by2 > mask.shape[0]:
                edge = True

        except:
            print(traceback.format_exc())

        return edge, [bx1, by1, bx2, by2]
    
    def populate_cell_dataset(self, masks, edge_cells=False):
        
        self.cell_dataset = []

        for mask_index, mask in enumerate(masks):

            mask_ids = np.unique(mask)

            for mask_id in mask_ids:

                try:

                    if mask_id != 0:

                        cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                        cell_mask[mask == mask_id] = 1

                        cnt, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cnt = cnt[0]

                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        x, y, w, h = cv2.boundingRect(cnt)

                        edge, crop_coords = self.check_edge_cell(cnt, mask)

                        x1,y1,x2,y2 = crop_coords
                        cell_mask_crop = cell_mask[y1:y2,x1:x2]

                        cnt, _ = cv2.findContours(cell_mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cnt = cnt[0]

                        if h > w:
                            vertical = True
                        else:
                            vertical = False

                        dat = {"mask_index":mask_index,
                               "mask_id":mask_id,
                               "edge":edge,
                               "bbox":[x1,y1,x2,y2],
                               "cell_centre":[cx,cy],
                               "cnt":cnt,
                               "cell_mask": cell_mask_crop,
                               "vertical":vertical,
                               }

                        if edge_cells == False:
                            if edge == False:
                                self.cell_dataset.append(dat)

                        else:
                            self.cell_dataset.append(dat)

                except:
                    pass
        
        if self.verbose:
            print(f"Imported {len(self.cell_dataset)} cells")
            
        return self.cell_dataset

    @staticmethod
    def get_mask_medial_axis(cell_mask, cell_contour = None,
            refine = True, iterations = 3):
        
        coords_list = []
        
        for i in range(iterations):
            skeleton = medial_axis(cell_mask, return_distance=False)
            coords = np.flip(np.transpose(np.nonzero(skeleton)),axis=1)
            coords_list.append(coords)
            
        coords = np.vstack(coords_list)
        coords = np.unique(coords, axis=0)
         
        polygon = Polygon(cell_contour.coords)

        centroid = polygon.centroid
        cell_radius = cell_contour.distance(centroid)

        if refine:
            coords = [p for p in coords if cell_contour.distance(Point(p)) > cell_radius*0.8]
            coords = np.array(coords)

        return coords, cell_radius

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

        if len(polygon_outline.coords) < 100:
            polygon_outline = BactFit.resize_line(polygon_outline, 200)
            polygon = Polygon(polygon_outline.coords)

        # Extract the exterior coordinates of the polygon
        exterior_coords = np.array(polygon.exterior.coords)

        # Compute the Voronoi diagram of the exterior coordinates
        vor = Voronoi(exterior_coords)

        # Function to check if a point is inside the polygon
        def point_in_polygon(point, polygon):
            return polygon.contains(Point(point))

        # Extract the medial axis points from the Voronoi vertices
        coords = [vor.vertices[i].tolist() for i, region in enumerate(vor.regions) if -1 not in region]
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
    def get_cell_contour(cnt, smooth = True):
        
        cell_contour_coords = cnt.reshape(-1,2)
        
        if smooth:
            cell_contour_coords = BactFit.moving_average(cell_contour_coords)
        
        cell_contour = LinearRing(cell_contour_coords)
        cell_contour = cell_contour.buffer(0.5)
        cell_contour = LinearRing(cell_contour.exterior.coords)
        cell_contour = BactFit.resize_line(cell_contour, 100)
        
        return cell_contour

    @staticmethod
    def bactfit_result(params, cell_polygon, poly_params,
            fit_mode = "directed_hausdorff", containment_penalty = False):
        
        x_min = params[0]
        x_max = params[1]
        cell_width = params[2]
        x_offset = params[3]
        y_offset = params[4]

        p = np.poly1d(poly_params)
        x_fitted = np.linspace(x_min, x_max, num=20)
        y_fitted = p(x_fitted)
        
        x_fitted += x_offset
        y_fitted += y_offset
    
        midline_coords = np.column_stack((x_fitted,y_fitted))
    
        midline = LineString(midline_coords)
    
        cell_fit = midline.buffer(cell_width)

        distance = BactFit.compute_bacfit_distance(cell_fit, cell_polygon,
            fit_mode, containment_penalty)

        x = midline_coords[:,0]
        y = midline_coords[:,1]
        bisector_coords = BactFit.get_poly_coords(x, y,
                                            poly_params, 
                                            n_points=100, 
                                            margin=cell_width*2)
        
        bisector = LineString(bisector_coords)
        
        fit_data = {"cell_fit":cell_fit,
                    "cell_midline":midline,
                    "cell_width":cell_width,
                    "bisector":bisector,
                    "error":distance,
                    }
 
        return fit_data

    @staticmethod
    def compute_bacfit_distance(midline_buffer, cell_polygon,
            fit_mode = "directed_hausdorff", containment_penalty = False):

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

            if containment_penalty:
                if cell_polygon.contains(midline_buffer) == False:
                    distance = distance * 1.5

        except:
            print(traceback.format_exc())
            distance = np.inf

        return distance


    @staticmethod
    def bactfit_function(params, cell_polygon, poly_params,
            fit_mode ="directed_hausdorff", containment_penalty=False):
        
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

            p = np.poly1d(poly_params)
            x_fitted = np.linspace(x_min, x_max, num=10)
            y_fitted = p(x_fitted)
            
            x_fitted += x_offset
            y_fitted += y_offset

            midline_coords = np.column_stack((x_fitted,y_fitted))
        
            midline = LineString(midline_coords)

            midline_buffer = midline.buffer(cell_width)

            distance = BactFit.compute_bacfit_distance(midline_buffer, cell_polygon,
                fit_mode, containment_penalty)

        except:
            print(traceback.format_exc())
            distance = np.inf

        return distance

    @staticmethod
    def get_poly_coords(x, y, coefficients, 
                        margin = 0, n_points=10):
        
        x1 = np.min(x) - margin
        x2 = np.max(x) + margin
        
        p = np.poly1d(coefficients)
        x_fitted = np.linspace(x1, x2, num=n_points)
        y_fitted = p(x_fitted)

        return np.column_stack((x_fitted,y_fitted))

    @staticmethod
    def fit_poly(coords, degree = 2, vertical = False,
                 constrained = True, 
                 constraining_points = [], 
                 minimise_curvature = False, maxiter = 50):
        
        def polynomial_fit(params, x):
            # Reverse the parameters to match np.polyfit order
            params = params[::-1]
            return sum(p * x**i for i, p in enumerate(params))

        def objective_function(params, x, y, minimise_curvature=True):
            
            fit_error = np.sum((polynomial_fit(params, x) - y)**2)
            
            if minimise_curvature:
                curvature_penalty = np.sum(np.diff(params, n=2)**2)
                
                fit_error = fit_error + curvature_penalty
            
            return fit_error

        def constraint_function(params, x_val, y_val):
            return polynomial_fit(params, x_val) - y_val

        def get_coords(x, y, coefficients, margin = 0, n_points=10):
            
            x1 = np.min(x) - margin
            x2 = np.max(x) + margin
            
            p = np.poly1d(coefficients)
            x_fitted = np.linspace(x1, x2, num=n_points)
            y_fitted = p(x_fitted)

            return np.column_stack((x_fitted,y_fitted))
            
        x = coords[:,0]
        y = coords[:,1]
        constraints = []
        
        param_list = []
        error_list = []
        success_list = []
        
        if constrained and len(constraining_points) > 0:
            
            for point in constraining_points:
                if len(point) == 2:
                    constraints.append({'type': 'eq', 
                                        'fun': constraint_function, 
                                        'args': point})
          
        if type(degree) != list:  
            degree = [degree]
            

        for deg in degree:
            
            params = np.polyfit(x, y, deg)
            
            result = minimize(objective_function, 
                              params, 
                              args=(x, y, minimise_curvature),
                              constraints=constraints, 
                              tol=1e-6, 
                              options={'maxiter': maxiter})
            
            param_list.append(result.x)
            error_list.append(result.fun)
            success_list.append(result.success)
                
        min_error_index = error_list.index(min(error_list))
        
        best_params = param_list[min_error_index]

        fitted_poly = get_coords(x, y, best_params)
            
        return fitted_poly, list(best_params)

    @staticmethod
    def register_line(line, cell_centre=[], vertical=False):

        if vertical:

            x_center = np.mean(line[:, 0])
            y_center = np.mean(line[:, 1])

            x = line[:,0] - x_center
            y = line[:,1] - y_center

            angle = math.radians(90)
            
            # Rotation matrix multiplication to get rotated x & y
            xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_center
            yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_center
            
            line[:, 0] = xr
            line[:, 1] = yr

        if len(cell_centre) == 2:

            x_center = np.mean(line[:, 0])
            y_center = np.mean(line[:, 1])

            line[:, 0] += cell_centre[0]
            line[:, 1] += cell_centre[1]

        return line

    @staticmethod
    def register_fit_data(fit_data, cell_centre = [], vertical = False):

        cell_fit = fit_data["cell_fit"]
        bisector = fit_data["bisector"]
        
        cell_fit = np.array(cell_fit.exterior.coords)
        bisector = np.array(bisector.coords)
        
        cell_fit = BactFit.register_line(cell_fit, cell_centre, vertical)
        bisector = BactFit.register_line(bisector, cell_centre, vertical)
        
        cell_fit = Polygon(cell_fit)
        bisector = LineString(bisector)
        
        fit_data["cell_fit"] = cell_fit
        fit_data["bisector"] = bisector

        return fit_data


    @staticmethod
    def rotate_polygon(polygon, angle = 90):

        polygon = shapely.affinity.rotate(polygon, angle=angle)

        return polygon

    @staticmethod
    def contour_fit(contour_data, progress_list,
            fit = True, fit_mode ="directed_hausdorff"):

        fit_data = None

        try:
            contour_index = contour_data["contour_index"]
            cell_polygon = contour_data["cell_polygon"]
            vertical = contour_data["vertical"]

            if vertical:
                cell_polygon = BactFit.rotate_polygon(cell_polygon)

            medial_axis_coords, radius = BactFit.get_polygon_medial_axis(cell_polygon)

            medial_axis_fit, poly_params = BactFit.fit_poly(medial_axis_coords,
                degree=[1, 2, 3], maxiter=50)

            x_min = np.min(medial_axis_fit[:, 0])
            x_max = np.max(medial_axis_fit[:, 0])
            x_offset = 0
            y_offset = 0
            params = [x_min, x_max, radius, x_offset, y_offset]

            if fit:
                result = minimize(BactFit.bactfit_function,
                    params,
                    args=(cell_polygon, poly_params, fit_mode),
                    tol=1e-12,
                    options={'maxiter': 100})

                params = result.x

            fit_result = BactFit.bactfit_result(params, cell_polygon, poly_params)
            fit_result = BactFit.register_fit_data(fit_result, vertical=vertical)

            fit_result["cell_fit"] = np.array(fit_result["cell_fit"].exterior.coords)
            fit_data = {"cell_fit":fit_result["cell_fit"],
                        "error":fit_result["error"],
                        }
        except:
            print(traceback.format_exc())
            pass

        progress_list.append(1)

        return fit_data

    @staticmethod
    def mask_fit(cell_data, progress_list = [],
            fit = True, fit_mode ="directed_hausdorff"):

        fit_data = None

        try:

            cell_mask = cell_data["cell_mask"]
            cell_centre = cell_data["cell_centre"]
            vertical = cell_data["vertical"]

            if vertical:
                cell_mask = cv2.rotate(cell_mask, cv2.ROTATE_90_CLOCKWISE)

            cnt, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = cnt[0]

            cell_contour = BactFit.get_cell_contour(cnt)

            medial_axis_coords, radius = BactFit.get_mask_medial_axis(cell_mask,
                cell_contour, iterations=1)

            medial_axis_fit, poly_params = BactFit.fit_poly(medial_axis_coords,
                degree=[1, 2], maxiter=50)

            x_min = np.min(medial_axis_fit[:, 0])
            x_max = np.max(medial_axis_fit[:, 0])
            x_offset = 0
            y_offset = 0
            params = [x_min, x_max, radius, x_offset, y_offset]

            if fit:
                result = minimize(BactFit.bactfit_function,
                    params,
                    args=(cell_contour, poly_params, fit_mode),
                    tol=1e-12,
                    options={'maxiter': 100})

                params = result.x

            fit_data = BactFit.bactfit_result(params, cell_contour, poly_params)
            fit_data = BactFit.register_fit_data(fit_data, cell_centre, vertical)

            fit_data["cell_fit"] = np.array(fit_data["cell_fit"].exterior.coords)
            fit_data["mask_id"] = cell_data["mask_id"]
            fit_data["mask_index"] = cell_data["mask_index"]

        except:
            pass

        progress_list.append(1)

        return fit_data

    def run_mask_fit(self, fit = True, parallel = False, max_workers = None,
            progress_callback = None, fit_mode = "directed_hausdorff"):

        fit_results = []

        if len(self.cell_dataset) > 0:

            with Manager() as manager:

                progress_list = manager.list()

                num_cells = len(self.cell_dataset)

                fit_func = partial(self.mask_fit,
                    progress_list=progress_list,
                    fit=fit, fit_mode=fit_mode)

                if parallel:

                    if max_workers == None:
                        max_workers = os.cpu_count()

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:

                        futures = [executor.submit(fit_func, cell_data) for cell_data in self.cell_dataset]

                        while any(not future.done() for future in futures):
                            progress = (sum(progress_list) / num_cells)
                            progress = progress * 100

                            if progress_callback is not None:
                                progress_callback.emit(progress)

                        fit_results = [future.result() for future in futures]

                else:

                    for cell_data in tqdm(self.cell_dataset):

                        fit_result = self.fit_cell(cell_data, progress_list, fit)

                        fit_results.append(fit_result)

                        progress = (sum(progress_list) / num_cells)
                        progress = progress * 100

                        if progress_callback is not None:
                            progress_callback.emit(progress)

        return fit_results


    def run_contour_fit(self, contour_dataset, fit = True, parallel = False, max_workers = None,
            progress_callback = None, fit_mode = "directed_hausdorff"):

        fit_results = []

        if len(contour_dataset) > 0:

            with Manager() as manager:

                progress_list = manager.list()

                num_cells = len(contour_dataset)

                fit_func = partial(BactFit.contour_fit,
                    progress_list=progress_list,
                    fit=fit, fit_mode=fit_mode)

                if parallel:

                    if max_workers == None:
                        max_workers = os.cpu_count()

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:

                        futures = [executor.submit(fit_func, cell_data) for cell_data in contour_dataset]

                        while any(not future.done() for future in futures):
                            progress = (sum(progress_list) / num_cells)
                            progress = progress * 100

                            if progress_callback is not None:
                                progress_callback.emit(progress)

                        fit_results = [future.result() for future in futures]

                else:

                    for cell_data in tqdm(contour_dataset):

                        fit_result = BactFit.contour_fit(cell_data, progress_list, fit)

                        fit_results.append(fit_result)

                        progress = (sum(progress_list) / num_cells)
                        progress = progress * 100

                        if progress_callback is not None:
                            progress_callback.emit(progress)

        return fit_results






                        