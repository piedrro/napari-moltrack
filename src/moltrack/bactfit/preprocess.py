from moltrack.bactfit.cell import Cell, CellList
from shapely.geometry import Polygon


def data_to_cells(segmentation_list, locs = None):

    cell_list = []

    for seg in segmentation_list:

        cell_images = {}

        if seg.shape[1] == 2:
            frame_index = -1

            cell_polygon = Polygon(seg)

        if seg.shape[1] == 3:
            frame_index = seg[0, 0]

            seg = seg[1:]

            cell_polygon = Polygon(seg)

        centroid = cell_polygon.centroid
        cell_centre = [centroid.x, centroid.y]

        minx, miny, maxx, maxy = cell_polygon.bounds

        bbox = [minx, miny, maxx, maxy]

        h = maxy - miny
        w = maxx - minx

        if h > w:
            vertical = True
        else:
            vertical = False

        cell_data = {
            "cell_polygon": cell_polygon,
            "cell_centre": cell_centre,
            "bbox": bbox,
            "height": h,
            "width": w,
            "vertical": vertical,
            "frame_index": frame_index
        }

        cell = Cell(cell_data)

        cell_list.append(cell)

    cell_list = CellList(cell_list)

    return cell_list



