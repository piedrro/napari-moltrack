from moltrack.bactfit.cell import Cell, CellList
from moltrack.bactfit.utils import resize_line, rotate_linestring, fit_poly, get_vertical, moving_average, get_polygon_midline
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import Voronoi
import numpy as np
import cv2



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

    if len(cell_list):
        cell_list = CellList(cell_list)

    return cell_list


def mask_to_cells(masks, images=None, locs=None):

    mask_list = None

    if isinstance(masks, np.ndarray):
        if len(masks.shape) == 3:
            mask_list = [mask for mask in masks]
        else:
            mask_list = [masks]
    if type(masks) == list:
        mask_list = masks

    if mask_list:

        cell_list = []

        for frame_index, mask in enumerate(mask_list):

            mask_ids = np.unique(mask)

            for mask_id in mask_ids:

                if mask_id == 0:
                    continue

                cell_mask = np.zeros_like(mask)
                cell_mask[mask == mask_id] = 255

                contours, _ = cv2.findContours(cell_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if len(contours) > 0:

                    contour = contours[0]
                    contour = contour.squeeze()

                    cell_polygon = Polygon(contour)

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
            
    if len(cell_list):
        cell_list = CellList(cell_list)

    return cell_list
