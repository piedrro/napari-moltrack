import pandas as pd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from bactfit.cell import ModelCell, Cell, CellList
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager

def create_cell_model(length=10, width=5, margin = 1):
    
    cell_model = {}
    
    x0 = y0 = width+margin
    
    # Define the coordinates of the line
    midline_x_coords = [x0, x0 + length]
    midline_y_coords = [y0, y0]
    midline_coords = list(zip(midline_x_coords, 
                              midline_y_coords))
    midline = LineString(midline_coords)
    
    polygon = midline.buffer(width)
    polygon_coords = np.array(polygon.exterior.coords)
    
    
    y0 = width+margin
    x0 = margin
    centerline_x_coords = [x0, x0 + length + (width*2)]
    centerline_y_coords = [y0, y0]
    centerline_coords = list(zip(centerline_x_coords, 
                                 centerline_y_coords))
    centerline_coords = np.array(centerline_coords)
    centerline = LineString(centerline_coords)
    
    centerline = BactFit.resize_line(centerline,100)
    
    cell_model['midline'] = midline
    cell_model['polygon'] = polygon
    cell_model['centerline'] = centerline
    cell_model["length"] = length
    cell_model["width"] = width

    return cell_model

def get_polylocs(polygons, locs):

    coords = np.stack([locs["x"], locs["y"]], axis=1)
    points = [Point(coord) for coord in coords]
    spatial_index = STRtree(points)
    
    polylocs = []
    
    for polygon_index, polygon in enumerate(polygons):
    
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
    
            polylocs.append(polygon_locs)
            
    if len(polylocs) > 0:
        polylocs = np.hstack(polylocs).view(np.recarray).copy()
        
    return polylocs




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


def get_render_jobs():
    
    total_locs = 0

    locs = pd.read_csv("cell_tracks.csv")
    locs = locs.to_records(index=False)
    
    with open('cell_shapes.json') as f:
        cell_shapes = json.load(f)
    
    cell_model = create_cell_model()
    
    polygons = [Polygon(coords) for coords in cell_shapes["polygon_coords"]]
    midlines = [LineString(coords) for coords in cell_shapes["midline_coords"]]
    widths = [width for width in cell_shapes["width"]]
    
    locs = get_polylocs(polygons, locs)
    
    polygon_indexes = np.unique(locs["cell_index"])
    
    render_jobs = []
    
    for index in polygon_indexes:
        
        polygon = polygons[index]
        midline = midlines[index]
        width = widths[index]
        polylocs = locs[locs["cell_index"] == index]
        
        render_jobs.append({"polygon": polygon,
                            "midline": midline,
                            "width": width,
                            "polylocs": polylocs,
                            "cell_model":cell_model})
        
        total_locs += len(polylocs)
    
    return render_jobs, total_locs
    
def split_linestring(linestring, num_segments):
    points = [linestring.interpolate(float(i) / num_segments, normalized=True) for i in range(num_segments + 1)]
    return [LineString([points[i], points[i + 1]]) for i in range(num_segments)]

def transform_locs(dat, progress_list=[], n_segments=1000, reflect = True):

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

        locs = dat["polylocs"]
        source_midline = dat["midline"]
        source_width = dat["width"]
        cell_model = dat["cell_model"]
        target_centerline = cell_model["centerline"]
        target_width = cell_model["width"]
        target_polygon = cell_model["polygon"]

        source_centerline = find_centerline(source_midline, source_width)
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
                
                # if distance > target_width:
                #     distance=target_width

                index_list = [closest_segment_index, abs(n_segments-closest_segment_index)]
                # index_list = [closest_segment_index]
                
                sign_list = [-1,1]
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

    except:
        print(traceback.format_exc())
        pass

    return transformed_locs



with open('cell_shapes.json') as f:
    cell_shapes = json.load(f)

locs = pd.read_csv("cell_tracks.csv")
locs = locs.to_records(index=False)



name_list = cell_shapes["name"]

cells = []

for cell_index, name in enumerate(name_list):
    
    cell_index = name_list.index(name)
    
    midline_coords = [""]
    
    cell_dict = {}
    for key,value in cell_shapes.items():
        cell_dict[key]=value[cell_index]
        
    cell_dict["cell_index"] = cell_index
        
    cell = Cell(cell_dict)
    cells.append(cell)

cells = CellList(cells)
    
cells.add_localisations(locs)
model = ModelCell(length=10, width=5)

# cells.transform_locs(model)
# cells.plot_cell_heatmap()
# cells.plot_cell_render()








# render_jobs, total_locs = get_render_jobs()

# dat = render_jobs[0]

# transformed_locs  = transform_locs(dat)


# transformed_locs = np.hstack(transformed_locs).view(np.recarray).copy()
# points = np.stack([transformed_locs["x"], transformed_locs["y"]], axis=1)
# polygon = render_jobs[0]["cell_model"]["polygon"]

# plt.scatter(points[:,0], points[:,1])
# plt.plot(*polygon.exterior.xy)
# plt.show()

# if __name__ == "__main__":
#
#     render_jobs, total_locs = get_render_jobs()
#
#     cpu_count = cpu_count()
#
#     transformed_locs = []
#
#     with Manager() as manager:
#
#         progress_list = manager.list()
#
#         with ProcessPoolExecutor() as executor:
#
#             futures = {executor.submit(transform_locs, dat, progress_list): dat for dat in render_jobs}
#
#             completed = 0
#             for future in as_completed(futures):
#                 tlocs = future.result()
#                 transformed_locs.append(tlocs)
#
#     transformed_locs = [loc for loc in transformed_locs if len(loc) > 0]
#     transformed_locs = np.hstack(transformed_locs).view(np.recarray).copy()
#
#     polygon = render_jobs[0]["cell_model"]["polygon"]
#     points = np.stack([transformed_locs["x"], transformed_locs["y"]], axis=1)
#
#     #create heatmap
#     heatmap, xedges, yedges = np.histogram2d(transformed_locs["x"], transformed_locs["y"], bins=30, density=False)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
#     plt.rcParams["axes.grid"] = False
#     plt.imshow(heatmap.T, extent=extent, origin='lower')
#     plt.plot(*polygon.exterior.xy)
    #remove tick lines
    

    # plt.scatter(points[:,0], points[:,1])
    # plt.plot(*polygon.exterior.xy)
    # plt.show()



# dat = render_jobs[0]
#
# transformed_locs = transform_locs(dat)










