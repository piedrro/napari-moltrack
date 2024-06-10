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

class oufti:

    def export_mesh(self, path, progress_callback=None):

        try:

            export_polygons = self.get_export_polygons()

            oufti_data = []

            for data in export_polygons:

                mesh_data = self.get_cell_mesh(data)

                oufti_data.append(mesh_data)

            if len(oufti_data) > 1:

                self.export_oufti(oufti_data,path)

                print("Exported to: ", path)

        except:
            print(traceback.format_exc())
            pass


    def get_cell_mesh(self, cell, n_segments = 50):

        mesh_data = None

        try:

            midline_coords = np.array(cell["midline_coords"])
            polygon_coords = np.array(cell["polygon_coords"])
            width = cell["width"]

            polygon = Polygon(polygon_coords)
            midline = LineString(midline_coords)

            centerline = oufti.find_centerline(midline,width,True)
            centerline_coords = np.array(centerline.coords)

            left_coords, right_coords = oufti.get_boundary_lines(centerline, polygon)

            mesh_data = oufti.get_mesh(left_coords, right_coords,
                centerline_coords, n_segments=n_segments)

        except:
            print(traceback.format_exc())
            pass

        return mesh_data

    @staticmethod
    def get_boundary_lines(centerline, polygon):
        try:
            polygon_coords = np.array(polygon.exterior.coords)

            start_point = centerline.coords[0]
            end_point = centerline.coords[-1]

            start_index = np.argmin(cdist(polygon_coords, [start_point]))
            end_index = np.argmin(cdist(polygon_coords, [end_point]))

            start_point = polygon_coords[start_index]
            end_point = polygon_coords[end_index]

            if end_index < start_index:
                length = len(polygon_coords) - start_index + end_index
            else:
                length = end_index - start_index

            # rotate polygon so that start point is at the beginning
            polygon_coords = np.concatenate([polygon_coords[start_index:],
                                             polygon_coords[:start_index]], axis=0)

            left_coords = polygon_coords[:length + 1]
            right_coords = polygon_coords[length:]
            right_coords = np.concatenate([right_coords, [polygon_coords[0]]], axis=0)

            # check start of left_coords is equal to start_point, else flip
            if not np.allclose(left_coords[0], start_point):
                left_coords = np.flip(left_coords, axis=0)
            if not np.allclose(right_coords[0], start_point):
                right_coords = np.flip(right_coords, axis=0)

        except:
            left_coords = None
            right_coords = None

        return left_coords, right_coords

    @staticmethod
    def get_mesh(left_coords, right_coords, centerline_coords,
            n_segments=50, bisector_length=100):

        mesh_data = None

        try:
            line_resolution = n_segments * 10
            centerline = LineString(centerline_coords)

            # resize left and right lines to have the same number of points
            right_line = LineString(right_coords)
            left_line = LineString(left_coords)
            right_line = BactFit.resize_line(right_line, line_resolution)
            left_line = BactFit.resize_line(left_line, line_resolution)
            right_coords = np.array(right_line.coords)
            left_coords = np.array(left_line.coords)

            # initialize lists for indices
            left_indices = []
            right_indices = []

            distances = np.linspace(0, centerline.length, n_segments)
            centerline_segments = [LineString([centerline.interpolate(distance - 0.01),
                                               centerline.interpolate(distance + 0.01)]) for distance in distances]

            centerline_segments = centerline_segments[1:-1]

            # iterate over centerline segments and find intersection with left and right lines
            for segment in centerline_segments:
                left_bisector = segment.parallel_offset(bisector_length, 'left')
                right_bisector = segment.parallel_offset(bisector_length, 'right')

                left_bisector = left_bisector.boundary.geoms[1]
                right_bisector = right_bisector.boundary.geoms[0]

                bisector = LineString([left_bisector, right_bisector])

                left_intersection = bisector.intersection(left_line)
                right_intersection = bisector.intersection(right_line)

                if left_intersection.geom_type == "Point" and right_intersection.geom_type == "Point":
                    left_index = np.argmin(cdist(left_coords, [left_intersection.coords[0]]))
                    right_index = np.argmin(cdist(right_coords, [right_intersection.coords[0]]))

                    # check if indices are already in list
                    if len(left_indices) == 0:
                        left_indices.append(left_index)
                        right_indices.append(right_index)
                    else:
                        # ensure that indices are always increasing
                        max_left = max(left_indices)
                        max_right = max(right_indices)

                        if max_left > left_index:
                            max_left += 1
                            if max_left > len(left_coords) - 2:
                                left_index = None
                            else:
                                left_index = max_left

                        if max_right > right_index:
                            max_right += 1
                            if max_left > len(right_coords) - 2:
                                right_index = None
                            else:
                                right_index = max_right

                        if left_index is not None and right_index is not None:
                            left_indices.append(left_index)
                            right_indices.append(right_index)

            # add start and end points to indices
            left_indices.insert(0, 0)
            right_indices.insert(0, 0)
            left_indices.append(len(left_coords) - 1)
            right_indices.append(len(right_coords) - 1)

            # get coordinates from indices
            left_coords = left_coords[np.array(left_indices)]
            right_coords = right_coords[np.array(right_indices)]

            #swap x and y
            left_coords = np.array([left_coords[:,1], left_coords[:,0]]).T
            right_coords = np.array([right_coords[:,1], right_coords[:,0]]).T

            mesh = np.hstack((left_coords, right_coords))
            model = np.vstack((left_coords, np.flipud(right_coords)))

            mesh = mesh + 1
            model = model + 1

            steplength, steparea, stepvolume = oufti.compute_line_metrics(mesh)

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

    @staticmethod
    def find_centerline(midline, width, smooth=True):
        try:
            def extract_end_points(line, num_points=2):
                coords = list(line.coords)
                if len(coords) < num_points:
                    raise ValueError("The LineString does not have enough points.")
                start_points = coords[:num_points]
                end_points = coords[-num_points:]
                return start_points, end_points

            def extend_away(points, distance, ):
                if len(points) < 2:
                    raise ValueError("At least two points are required to determine the direction for extension.")

                p1 = Point(points[0])
                p2 = Point(points[1])

                dx = p2.x - p1.x
                dy = p2.y - p1.y
                length = np.hypot(dx, dy)
                factor = distance / length

                # Extend p1 away from p2
                extended_x1 = p1.x - factor * dx
                extended_y1 = p1.y - factor * dy

                # Similarly for the other end
                p3 = Point(points[-1])
                p4 = Point(points[-2])

                dx_end = p4.x - p3.x
                dy_end = p4.y - p3.y
                length_end = np.hypot(dx_end, dy_end)
                factor_end = distance / length_end

                # Extend p3 away from p4
                extended_x2 = p3.x - factor_end * dx_end
                extended_y2 = p3.y - factor_end * dy_end

                return (extended_x1, extended_y1), (extended_x2, extended_y2)

            model = midline.buffer(width)

            centerline = BactFit.resize_line(midline, 1000)  # High resolution with 1000 points

            start_points, end_points = extract_end_points(centerline)

            extension_distance = width * 3

            extended_start = extend_away(start_points, extension_distance)
            extended_end = extend_away(end_points, extension_distance)

            extended_start_line = LineString([start_points[0], extended_start[0]])
            extended_end_line = LineString([end_points[-1], extended_end[1]])

            outline = LineString(model.exterior.coords)
            intersections_start = outline.intersection(extended_start_line).coords[0]
            intersections_end = outline.intersection(extended_end_line).coords[0]

            centerline_coords = np.array(centerline.coords)
            centerline_coords = np.insert(centerline_coords, 0, intersections_start, axis=0)
            centerline_coords = np.append(centerline_coords, [intersections_end], axis=0)
            centerline = LineString(centerline_coords)

            if smooth:
                vertical = BactFit.get_vertical(model)

                if vertical:
                    centerline = BactFit.rotate_linestring(centerline, angle=90)

                centerline_coords = np.array(centerline.coords)
                constraining_points = [centerline_coords[0], centerline_coords[-1]]

                centerline_coords, _ = BactFit.fit_poly(centerline_coords, degree=[1, 2, 3],
                    maxiter=100, constraining_points=constraining_points, constrained=True)

                centerline = LineString(centerline_coords)

                if vertical:
                    centerline = BactFit.rotate_linestring(centerline, angle=-90)

        except:
            print(traceback.format_exc())
            pass

        return centerline


    def find_closest_point(self, point, line):
        point = Point(point)

        pol_ext = LinearRing(line)
        d = pol_ext.project(point)
        p = pol_ext.interpolate(d)
        closet_point = list(p.coords)[0]

        return closet_point

    @staticmethod
    def euclidian_distance(x1, y1, x2, y2):
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        return distance

    @staticmethod
    def polyarea( x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @staticmethod
    def compute_line_metrics(mesh):
        steplength = oufti.euclidian_distance(mesh[1:, 0] + mesh[1:, 2], mesh[1:, 1] + mesh[1:, 3], mesh[:-1, 0] + mesh[:-1, 2], mesh[:-1, 1] + mesh[:-1, 3]) / 2

        steparea = []
        for i in range(len(mesh) - 1):
            steparea.append(oufti.polyarea([*mesh[i:i + 2, 0], *mesh[i:i + 2, 2][::-1]],
                [*mesh[i:i + 2, 1], *mesh[i:i + 2, 3][::-1]]))

        steparea = np.array(steparea)

        d = oufti.euclidian_distance(mesh[:, 0], mesh[:, 1], mesh[:, 2], mesh[:, 3])
        stepvolume = (d[:-1] * d[1:] + (d[:-1] - d[1:]) ** 2 / 3) * steplength * math.pi / 4

        return steplength, steparea, stepvolume

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