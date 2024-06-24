import numpy as np
import traceback
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from moltrack.bactfit.utils import resize_line, rotate_linestring, fit_poly, get_vertical
import matplotlib.pyplot as plt
import math

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

def calculate_angle(segment, point):
    # Calculate the angle between the segment and the point
    line_start = segment.coords[0]
    line_end = segment.coords[1]
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    segment_angle = math.atan2(dy, dx)
    point_angle = math.atan2(point.y - line_start[1], point.x - line_start[0])
    angle = point_angle - segment_angle
    return math.degrees(angle)

def calculate_new_point(segment, distance, angle):
     line_start = segment.coords[0]
     line_end = segment.coords[1]
     segment_angle = math.atan2(line_end[1] - line_start[1], line_end[0] - line_start[0])
     new_angle = segment_angle + angle
     new_x = line_start[0] + distance * math.cos(new_angle)
     new_y = line_start[1] + distance * math.sin(new_angle)
     return [new_x, new_y]

def reflect_loc_horizontally(loc, centroid):

    center_y = centroid[1]
    centroid_distance = loc.y - center_y

    if centroid_distance < 0:
        loc.y = loc.y + abs(centroid_distance)*2
    else:
        loc.y = loc.y - abs(centroid_distance)*2

    return loc

def reflect_loc_vertically(loc, centroid):

    center_x = centroid[0]
    centroid_distance = loc.x - center_x

    if centroid_distance < 0:
        loc.x = loc.x + abs(centroid_distance)*2
    else:
        loc.x = loc.x - abs(centroid_distance)*2

    return loc
        

def cell_coordinate_transformation(cell, target_cell,
        n_segments=1000, progress_list = []):
    

    try:

        locs = cell.locs

        source_polygon = cell.cell_fit
        source_midline = cell.cell_midline
        
        source_width = cell.cell_width
        target_width = target_cell.cell_length
        
        target_midline = target_cell.cell_midline
        target_polygon = target_cell.cell_polygon
        
        source_segments = split_linestring(source_midline, n_segments)
        target_segments = split_linestring(target_midline, n_segments)
        
        # Create STRtree for segments
        tree = STRtree(source_segments)
        
        transformed_locs = []
        
        for loc in locs:

            try:

                point = Point(loc["x"], loc["y"])
                
                closest_segment_index = tree.nearest(point)
                nearest_segment = source_segments[closest_segment_index]
                
                source_distance = point.distance(nearest_segment)
                angle = calculate_angle(nearest_segment, point)
                
                target_distance = target_width * (source_distance / source_width)
                
                # Use the corresponding target segment
                target_segment = target_segments[closest_segment_index]
                
                # Calculate the new point in the target segment
                new_point_coords = calculate_new_point(target_segment, target_distance, angle)
                
                if target_polygon.contains(Point(new_point_coords)):
                    
                    if len(new_point_coords) == 0:
                        continue
                
                    tloc = loc.copy()
                    tloc["x"] = new_point_coords[0]
                    tloc["y"] = new_point_coords[1]

                    transformed_locs.append(tloc)
                
            except:
                print(traceback.format_exc())
                pass
            
        if len(transformed_locs) > 0:

            transformed_locs = np.hstack(transformed_locs).view(np.recarray).copy()

            cell.locs = transformed_locs
            cell.cell_polygon = target_polygon
            cell.cell_midline = target_midline
            
        else:
            cell.locs = None

    except:
        print(traceback.format_exc())
        pass

    return cell

