from shapely.geometry import Polygon, LineString
from moltrack.bactfit.fit import BactFit
import matplotlib.pyplot as plt
import traceback
import numpy as np
import shapely
from scipy.spatial import distance
from shapely.ops import nearest_points, split
from scipy.spatial.distance import cdist
from shapely.geometry import Point, LinearRing
from shapely.geometry.polygon import orient
import math
import os
import scipy

class _oufti_utils:

    def export_mesh_finished(self):

        self.update_ui()

    def export_mesh(self, path, progress_callback=None):

        try:

            export_polygons = self.get_export_polygons()

            oufti_data = []

            for data in export_polygons:

                mesh_data = self.get_cell_mesh(data)

                oufti_data.append(mesh_data)

                print(True)

            if len(oufti_data) > 1:

                self.export_oufti(oufti_data,path)

                print("Exported to: ", path)

        except:
            print(traceback.format_exc())
            pass


    def get_cell_mesh(self, cell, refit = True):

        mesh_data = None

        try:

            midline_coords = np.array(cell["midline_coords"])
            polygon_coords = np.array(cell["polygon_coords"])
            width = cell["width"]
            poly_params = cell["poly_params"]

            polygon = Polygon(polygon_coords)
            midline = LineString(midline_coords)

            vertical = self.get_vertical(polygon)

            if vertical:
                polygon = self.rotate_polygon(polygon, angle=90)

            if refit:

                if vertical:
                    midline = self.rotate_polygon(midline, angle=90)
                    midline_coords = np.array(midline.coords)

                constraining_points = [midline_coords[0], midline_coords[-1]]
                midline_coords, poly_params = BactFit.fit_poly(midline_coords, degree=[1, 2, 3], maxiter=100, minimise_curvature=True,
                    constraining_points=constraining_points, constrained=True)
                midline = LineString(midline_coords)

            bisector_coords = BactFit.get_poly_coords(
                midline_coords[:,0],
                midline_coords[:,1],
                poly_params,
                n_points=100,
                margin=width*3)

            bisector = LineString(bisector_coords)

            if vertical:
                bisector = self.rotate_polygon(bisector, angle=-90)
                polygon = self.rotate_polygon(polygon, angle=-90)

            bisector_coords = np.array(bisector.coords)
            polygon_coords = np.array(polygon.exterior.coords)

            #flip x and y
            polygon_coords = np.flip(polygon_coords, axis=1)
            bisector_coords = np.flip(bisector_coords, axis=1)
            polygon = Polygon(polygon_coords)
            bisector = LineString(bisector_coords)

            midline = self.get_mid_line(polygon, bisector)

            if midline is None:
                return None

            left_coords, right_coords, midline_coords = self.get_boundary_lines(midline, polygon)

            midline_coords, end_intersections = self.trim_midline(left_coords,
                right_coords, midline_coords, margin=10)

            mesh_data = self.get_mesh_data(left_coords,right_coords,
                midline_coords, end_intersections)

        except:
            print(traceback.format_exc())
            pass

        return mesh_data


    def get_mesh_data(self, left_coords, right_coords, midline_coords,
            end_intersections, bisector_length=100, n_segments=20):

        mesh_data = {}

        try:

            left_line = LineString(left_coords)
            right_line = LineString(right_coords)

            left_line = self.resize_line(left_line, n_segments)
            right_line = self.resize_line(right_line, n_segments)

            midline = LineString(midline_coords)
            midline = self.resize_line(midline, n_segments)

            distances = np.linspace(0, midline.length, n_segments)[1:]

            mid_line_segments = [LineString([midline.interpolate(distance - 0.01),
                                             midline.interpolate(distance + 0.01)]) for distance in distances]

            right_line_data = [end_intersections[0].tolist()]
            left_line_data = [end_intersections[0].tolist()]

            for segment in mid_line_segments:
                left_bisector = segment.parallel_offset(bisector_length, 'left')
                right_bisector = segment.parallel_offset(bisector_length, 'right')

                left_bisector = left_bisector.boundary.geoms[1]
                right_bisector = right_bisector.boundary.geoms[0]

                bisector = LineString([left_bisector, right_bisector])

                left_intersection = bisector.intersection(left_line)
                right_intersection = bisector.intersection(right_line)

                if left_intersection.geom_type == "Point" and right_intersection.geom_type == "Point":
                    right_line_data.append(np.array(left_intersection.xy).reshape(2).tolist())
                    left_line_data.append(np.array(right_intersection.xy).reshape(2).tolist())

            right_line_data.append(end_intersections[-1].tolist())
            left_line_data.append(end_intersections[-1].tolist())

            left_line_data = np.array(left_line_data)
            right_line_data = np.array(right_line_data)

            mesh = np.hstack((left_line_data, right_line_data))
            model = np.vstack((left_line_data, np.flipud(right_line_data)))

            mesh = mesh + 1
            model = model + 1

            steplength, steparea, stepvolume = self.compute_line_metrics(mesh)

            polygon = Polygon(model)
            polygon = orient(polygon)

            boundingbox = np.asarray(polygon.bounds)

            boundingbox[0:2] = np.floor(boundingbox[0:2])
            boundingbox[2:4] = np.ceil(boundingbox[2:4])
            boundingbox[2:4] = boundingbox[2:4] - boundingbox[0:2]
            boundingbox = boundingbox.astype(float)

            mesh_data = {"mesh": mesh, "model":model, "steplength":steplength,
                         "steparea":steparea, "stepvolume":stepvolume, "boundingbox":boundingbox}

        except:
            print(traceback.format_exc())

        return mesh_data

    def resize_line(self, line, length):
        distances = np.linspace(0, line.length, length)
        line = LineString([line.interpolate(distance) for distance in distances])

        return line

    def get_mid_line(self, polygon, bisector):

        splitted = shapely.ops.split(polygon, bisector)

        midline = None

        if len(splitted.geoms) == 2:

            intersecting_points = bisector.intersection(polygon)
            intersecting_points = np.array(intersecting_points.coords)

            first_point = intersecting_points[0]
            last_point = intersecting_points[-1]

            polygon_coords = np.array(polygon.exterior.coords)
            first_index = np.argmin(distance.cdist([first_point], polygon_coords)[0])
            last_index = np.argmin(distance.cdist([last_point], polygon_coords)[0])

            if first_index > last_index:
                first_index, last_index = last_index, first_index

            # Rotate the cell model coordinates so that the first index is the first point
            polygon_coords = np.roll(polygon_coords, -first_index, axis=0)

            # If the polygon was closed, ensure the first and last points are the same
            if (polygon_coords[0] != polygon_coords[-1]).any():
                polygon_coords = np.vstack([polygon_coords, polygon_coords[0]])

            split_index = last_index - first_index

            left_line = polygon_coords[:split_index + 1]
            right_line = polygon_coords[split_index:]

            right_line = np.flipud(right_line)

            right_line = self.resize_line(LineString(right_line), 100)
            left_line = self.resize_line(LineString(left_line), 100)

            left_coords = np.array(left_line.coords)
            right_coords = np.array(right_line.coords)

            midline = (left_coords + right_coords) / 2
            midline = LineString(midline)

        return midline

    def get_boundary_lines(self, midline, polygon, smooth=True, n_segments=100):

        try:

            intersect_splitter = midline.intersection(polygon)
            geomcollect = split(polygon, midline)
            left_coords, right_coords = geomcollect.geoms[0], geomcollect.geoms[1]

            left_coords = self.remove_intersecting(left_coords, midline)
            right_coords = self.remove_intersecting(right_coords, midline)

            distances = np.min(cdist(left_coords, right_coords), axis=0)
            distances_flip = np.min(cdist(left_coords, np.flip(right_coords)), axis=0)

            distances = np.sum(np.take(distances, [0, -1]))
            distances_flip = np.sum(np.take(distances_flip, [0, -1]))

            if distances_flip > distances:
                right_coords = np.flip(right_coords, axis=0)

            p1 = (right_coords[0] + left_coords[0]) / 2
            p2 = (right_coords[-1] + left_coords[-1]) / 2

            p1 = self.find_closest_point(p1, midline)
            p2 = self.find_closest_point(p2, midline)

            left_coords = np.concatenate(([p1], left_coords, [p2]))
            right_coords = np.concatenate(([p1], right_coords, [p2]))

            right_coords = np.flip(right_coords, axis=0)

            left_line = LineString(left_coords)
            right_line = LineString(right_coords)

            left_line = self.resize_line(left_line, n_segments)
            right_line = self.resize_line(right_line, n_segments)

            left_coords = np.array(left_line.xy).T
            right_coords = np.array(right_line.xy).T
            midline_coords = (left_coords + np.flipud(right_coords)) / 2

        except:
            left_coords, right_coords, midline_coords = None, None, None

        return left_coords, right_coords, midline_coords

    def remove_intersecting(self, line, intersecting_line):

        line = LineString(line.exterior)

        intersection = line.intersection(intersecting_line)
        intersection = np.array([[geom.xy[0][0], geom.xy[1][0]] for geom in intersection.geoms])

        line = np.array(line.xy).T

        distance = cdist(line, intersection)
        end_indexes = sorted([np.argmin(dist).tolist() for dist in distance.T])

        end_indexes = np.unique(end_indexes).tolist()

        if end_indexes[1] - end_indexes[0] > 1:
            line = np.roll(line, -end_indexes[1], 0)
            distance = cdist(line, intersection)
            end_indexes = sorted([np.argmin(dist).tolist() for dist in distance.T])

        overlap_length = abs(end_indexes[0] - end_indexes[-1])

        line = np.roll(line, -end_indexes[0], 0)
        line = line[overlap_length:]

        distances = cdist(line, np.array(intersecting_line.xy).T)
        distances = np.min(distances, axis=1)
        del_indexes = np.argwhere(distances < 0.5).flatten()

        line = np.delete(line, del_indexes, axis=0)

        return line

    def find_closest_point(self, point, line):
        point = Point(point)

        pol_ext = LinearRing(line)
        d = pol_ext.project(point)
        p = pol_ext.interpolate(d)
        closet_point = list(p.coords)[0]

        return closet_point

    def euclidian_distance(self, x1, y1, x2, y2):
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        return distance

    def polyarea(self, x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def compute_line_metrics(self, mesh):
        steplength = self.euclidian_distance(mesh[1:, 0] + mesh[1:, 2], mesh[1:, 1] + mesh[1:, 3], mesh[:-1, 0] + mesh[:-1, 2], mesh[:-1, 1] + mesh[:-1, 3]) / 2

        steparea = []
        for i in range(len(mesh) - 1):
            steparea.append(self.polyarea([*mesh[i:i + 2, 0], *mesh[i:i + 2, 2][::-1]], [*mesh[i:i + 2, 1], *mesh[i:i + 2, 3][::-1]]))

        steparea = np.array(steparea)

        d = self.euclidian_distance(mesh[:, 0], mesh[:, 1], mesh[:, 2], mesh[:, 3])
        stepvolume = (d[:-1] * d[1:] + (d[:-1] - d[1:]) ** 2 / 3) * steplength * math.pi / 4

        return steplength, steparea, stepvolume

    def trim_midline(self, left_coords, right_coords, midline_coords, margin=10):
        try:
            start_point = left_coords[0]
            end_point = left_coords[-1]

            start_index = np.argmin(cdist([start_point], midline_coords))
            end_index = np.argmin(cdist([end_point], midline_coords))

            if start_index > end_index:
                start_index, end_index = end_index, start_index

            end_intersections = [midline_coords[start_index], midline_coords[end_index]]

            margin = 10

            if start_index >= margin:
                start_index -= margin
            if end_index <= len(midline_coords) + margin:
                end_index += margin

            midline_coords = midline_coords[start_index:end_index]

        except:
            pass

        return midline_coords, end_intersections

    def get_vertical(self, polygon):

        minx, miny, maxx, maxy = polygon.bounds

        h = maxy - miny
        w = maxx - minx

        if h > w:
            vertical = True
        else:
            vertical = False

        return vertical

    def rotate_polygon(self, polygon, angle=90):
        origin = polygon.centroid.coords[0]
        polygon = shapely.affinity.rotate(polygon, angle=angle, origin=origin)

        return polygon

    def export_oufti(self, oufti_data, file_path):

        file_path = os.path.splitext(file_path)[0] + ".mat"

        cell_data = []

        for dat in oufti_data:
            try:

                if dat is None:
                    continue

                cell_struct = {'mesh': dat["mesh"], 'model': dat["model"], 'birthframe': 1, 'divisions': [],
                               'ancestors': [], 'descendants': [], 'timelapse': False,
                               'algorithm': 5, 'polarity': 0, 'stage': 1, 'box': dat["boundingbox"],
                               'steplength': dat["steplength"], 'length': np.sum(dat["steplength"]),
                               'lengthvector': dat["steplength"], 'steparea': dat["steparea"], 'area': np.sum(dat["steparea"]),
                               'stepvolume': dat["stepvolume"].T, 'volume': np.sum(dat["stepvolume"])}

                cell_data.append(cell_struct)

            except:
                print(traceback.format_exc())
                pass

        cellListN = len(cell_data)
        cellList = np.zeros((1,), dtype=object)
        cellList_items = np.zeros((1, cellListN), dtype=object)

        microbeTrackerParamsString = "% This file contains MicrobeTracker settings optimized for wildtype E. coli cells at 0.114 um/pixel resolution (using algorithm 4)\n\nalgorithm = 4\n\n% Pixel-based parameters\nareaMin = 120\nareaMax = 2200\nthresFactorM = 1\nthresFactorF = 1\nsplitregions = 1\nedgemode = logvalley\nedgeSigmaL = 3\nedveSigmaV = 1\nvalleythresh1 = 0\nvalleythresh2 = 1\nerodeNum = 1\nopennum = 0\nthreshminlevel = 0.02\n\n% Constraint parameters\nfmeshstep = 1\ncellwidth =6.5\nfsmooth = 18\nimageforce = 4\nwspringconst = 0.3\nrigidityRange = 2.5\nrigidity = 1\nrigidityRangeB = 8\nrigidityB = 5\nattrCoeff = 0.1\nrepCoeff = 0.3\nattrRegion = 4\nhoralign = 0.2\neqaldist = 2.5\n\n% Image force parameters\nfitqualitymax = 0.5\nforceWeights = 0.25 0.5 0.25\ndmapThres = 2\ndmapPower = 2\ngradSmoothArea = 0.5\nrepArea = 0.9\nattrPower = 4\nneighRep = 0.15\n\n% Mesh creation parameters\nroiBorder = 20.5\nnoCellBorder = 5\nmaxmesh = 1000\nmaxCellNumber = 2000\nmaxRegNumber = 10000\nmeshStep = 1\nmeshTolerance = 0.01\n\n% Fitting parameters\nfitConvLevel = 0.0001\nfitMaxIter = 500\nmoveall = 0.1\nfitStep = 0.2\nfitStepM = 0.6\n\n% Joining and splitting\nsplitThreshold = 0.35\njoindist = 5\njoinangle = 0.8\njoinWhenReuse = 0\nsplit1 = 0\n\n% Other\nbgrErodeNum = 5\nsgnResize = 1\naligndepth = 1"

        for i in range(len(cell_data)):
            cellList_items[0, i] = cell_data[i]

        cellList[0] = cellList_items

        p = [];
        paramString = np.empty((len(microbeTrackerParamsString.split('\n')), 1), dtype=object)
        paramSplit = microbeTrackerParamsString.split('\n')
        for p_index in range(len(microbeTrackerParamsString.split('\n'))):
            paramString[p_index] = paramSplit[p_index]

        outdict = {'cellList': cellList, 'cellListN': cellListN,
                   'coefPCA': [], 'mCell': [], 'p': [], 'paramString': paramString,
                   'rawPhaseFolder': [], 'shiftfluo': np.zeros((2, 2)),
                   'shiftframes': [], 'weights': []}

        scipy.io.savemat(file_path, outdict)