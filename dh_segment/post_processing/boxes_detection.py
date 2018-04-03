import cv2
import numpy as np
from scipy.spatial import KDTree


def find_boxes(predictions: np.array, mode: str='min_rectangle', min_area: float=0.2,
               p_arc_length: float=0.01, n_max_boxes=1):
    """

    :param predictions: Uint8 binary 2D array
    :param mode: 'min_rectangle', 'quadrilateral', rectangle
    :param min_area:
    :param p_arc_length: when 'qualidrateral' mode is chosen
    :param n_max_boxes:
    :return: n_max_boxes of 4 corners [[x1,y2], [x2,y2], ... ]
    """
    _, contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        print('No contour found')
        return None
    # found_box = None
    found_boxes = list()

    h_img, w_img = predictions.shape[:2]

    def validate_box(box: np.array) -> (np.array, float):
        # Aproximate area computation (TODO eventually make a proper area computation)
        approx_area = (np.max(box[:, 0]) - np.min(box[:, 0])) * (np.max(box[:, 1] - np.min(box[:, 1])))
        # if approx_area > min_area * predictions.size and approx_area > biggest_area:
        if approx_area > min_area * predictions.size:

            # Correct out of range corners
            box = np.maximum(box, 0)
            box = np.stack((np.minimum(box[:, 0], predictions.shape[1]),
                            np.minimum(box[:, 1], predictions.shape[0])), axis=1)

            # return box
            return box, approx_area

    if mode not in ['quadrilateral', 'min_rectangle', 'rectangle']:
        raise NotImplementedError
    if mode == 'quadrilateral':
        for c in contours:
            epsilon = p_arc_length * cv2.arcLength(c, True)
            cnt = cv2.approxPolyDP(c, epsilon, True)
            # box = np.vstack(simplify_douglas_peucker(cnt[:, 0, :], 4))

            # Find extreme points in Convex Hull
            hull_points = cv2.convexHull(cnt, returnPoints=True)
            # points = cnt
            points = hull_points
            if len(points) > 4:
                # Find closes points to corner using nearest neighbors
                tree = KDTree(points[:, 0, :])
                _, ul = tree.query((0, 0))
                _, ur = tree.query((w_img, 0))
                _, dl = tree.query((0, h_img))
                _, dr = tree.query((w_img, h_img))
                box = np.vstack([points[ul, 0, :], points[ur, 0, :],
                                 points[dr, 0, :], points[dl, 0, :]])
            elif len(hull_points) == 4:
                box = hull_points[:, 0, :]
            else:
                    continue
            # Todo : test if it looks like a rectangle (2 sides must be more or less parallel)
            # todo : (otherwise we may end with strange quadrilaterals)
            if len(box) != 4:
                mode = 'min_rectangle'
                print('Quadrilateral has {} points. Switching to minimal rectangle mode'.format(len(box)))
            else:
                # found_box = validate_box(box)
                found_boxes.append(validate_box(box))
    if mode == 'min_rectangle':
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            # found_box = validate_box(box)
            found_boxes.append(validate_box(box))
    elif mode == 'rectangle':
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            # found_box = validate_box(box)
            found_boxes.append(validate_box(box))
    # sort by area
    found_boxes = [fb for fb in found_boxes if fb is not None]
    found_boxes = sorted(found_boxes, key=lambda x: x[1], reverse=True)
    if n_max_boxes == 1:
        if found_boxes != []:
            return found_boxes[0][0]
        else:
            return None
    else:
        return [fb[0] for i, fb in enumerate(found_boxes) if i <= n_max_boxes]