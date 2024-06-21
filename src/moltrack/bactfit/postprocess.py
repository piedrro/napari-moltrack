import numpy as np
import traceback
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from moltrack.bactfit.utils import resize_line, rotate_linestring, fit_poly, get_vertical
import matplotlib.pyplot as plt

def find_centerline(midline, width, smooth=True):

    centerline = None

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

        centerline = resize_line(midline, 1000)  # High resolution with 1000 points

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
            vertical = get_vertical(model)

            if vertical:
                centerline = rotate_linestring(centerline, angle=90)

            centerline_coords = np.array(centerline.coords)
            constraining_points = [centerline_coords[0], centerline_coords[-1]]

            centerline_coords, _ = fit_poly(centerline_coords, degree=[1, 2, 3], maxiter=100, constraining_points=constraining_points, constrained=True)

            centerline = LineString(centerline_coords)

            if vertical:
                centerline = rotate_linestring(centerline, angle=-90)

    except:
        print(traceback.format_exc())
        pass

    return centerline


def split_linestring(linestring, num_segments):
    points = [linestring.interpolate(float(i) / num_segments, normalized=True) for i in range(num_segments + 1)]
    return [LineString([points[i], points[i + 1]]) for i in range(num_segments)]

def cell_coordinate_transformation(cell, target_cell, n_segments=1000, reflect = True, progress_list = []):

    transformed_locs = []

    def compute_vectors(segments):
        unit_vectors = []
        perpendicular_vectors = []
        for segment in segments:
            segment_start = np.array(segment.coords[0])
            segment_end = np.array(segment.coords[1])
            segment_vector = segment_end - segment_start
            segment_length = np.linalg.norm(segment_vector)
            unit_vector = segment_vector / segment_length
            perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]])
            unit_vectors.append(unit_vector)
            perpendicular_vectors.append(perpendicular_vector)
        return unit_vectors, perpendicular_vectors

    try:

        locs = cell.locs

        source_polygon = cell.cell_polygon
        source_midline = cell.cell_midline
        source_width = cell.width

        # polygon_coords = np.array(source_polygon.exterior.coords)
        # coords = np.array([(loc["x"], loc["y"]) for loc in locs])
        # plt.plot(*polygon_coords.T)
        # plt.scatter(*coords.T)
        # plt.show()


        if cell.cell_centerline is None:
            cell.cell_centerline = find_centerline(source_midline, source_width)
            source_centerline = cell.cell_centerline

        target_width = target_cell.width
        target_centerline = target_cell.cell_centerline
        target_polygon = target_cell.cell_polygon

        source_segments = split_linestring(source_centerline, n_segments)
        target_segments = split_linestring(target_centerline, n_segments)

        # Precompute vectors for source and target segments
        source_unit_vectors, source_perpendicular_vectors = compute_vectors(source_segments)
        target_unit_vectors, target_perpendicular_vectors = compute_vectors(target_segments)

        # Create STRtree for segments
        tree = STRtree(source_segments)

        for loc in locs:

            try:

                point = Point(loc["x"], loc["y"])

                # Find the nearest segment to each point
                closest_segment_index = tree.nearest(point)

                nearest_segment = source_segments[closest_segment_index]

                segment_start = np.array(nearest_segment.coords[0])
                segment_vector = source_unit_vectors[closest_segment_index]
                point_vector = np.array([loc["x"], loc["y"]]) - segment_start

                # Compute the signed distance from the point to the nearest segment
                distance = point.distance(nearest_segment)
                distance_sign = np.sign(np.cross(segment_vector, point_vector))

                index_list = [closest_segment_index, abs(n_segments -closest_segment_index)]
                # index_list = [closest_segment_index]

                sign_list = [-1 ,1]
                # sign_list = [distance_sign]

                for segment_index in index_list:

                    for sign in sign_list:

                        signed_distance = distance*sign
                        # Compute the new distance in the target coordinate system
                        new_distance = target_width * (signed_distance / source_width)

                        # Calculate the new coordinates by moving perpendicular to the target segment at the new distance
                        segment_start_target = np.array(target_segments[segment_index].coords[0])
                        perp_vector = target_perpendicular_vectors[segment_index]
                        new_point_coords = segment_start_target + perp_vector * new_distance

                        if target_polygon.contains(Point(new_point_coords)):

                            tloc = loc.copy()
                            tloc["x"] = new_point_coords[0]
                            tloc["y"] = new_point_coords[1]

                            transformed_locs.append(tloc)

            except:
                pass

            progress_list.append(1)

        if len(transformed_locs) > 0:
            transformed_locs = np.hstack(transformed_locs).view(np.recarray).copy()

            cell.locs = transformed_locs
            cell.cell_polygon = target_polygon


        else:
            cell.locs = None

    except:
        print(traceback.format_exc())
        pass

    return cell