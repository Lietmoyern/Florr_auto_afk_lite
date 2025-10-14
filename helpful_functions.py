import cv2
import numpy as np
from scipy.spatial import distance
from rdp import rdp
from heapq import heappop, heappush
import pyautogui
import torch
from scipy.ndimage import distance_transform_edt
import json
from skimage.morphology import skeletonize as skimage_skeletonize
from scipy.spatial import KDTree
import random
import time
from detector import get_masks_by_iou, detect_afk_things
from ultralytics import YOLO


DEFAULT_COLOR_TOLERANCE = 40
DEFAULT_RDP_EPSILON = "0.2173913043478*width+0.4782608695652"
DEFAULT_EXTEND_LENGTH = 30
DEFAULT_KEY_PRESS_DELAY = 0.15
model_path = "./models/afk-det.pt"
afk_det_model = YOLO(model_path)
seg_model_path = "./models/afk-seg.pt"
afk_seg_model = YOLO(seg_model_path)

class AFK_BW:
    def __init__(self, image):
        self.offset = [0, 0]
        self.raw_image = image.copy()
        self.image = None
        self.start_p = None
        self.end_p = None
        self.length: float = None
        self.difficulty: float = None
        self.width: float = None
        self.extend_point: tuple[int, int] = None
        self.sorted_points: list[tuple[int, int]] = None
        self.rdp_points: list[tuple[int, int]] = None
        self.extend_point: tuple[int, int] = None
        self.white_labels = None
        self.max_white_label = None
        self.difficulty = None
        self.start_color = (0, 0, 0)
        self.inverse_start_color = (255, 255, 255)

    def crop_nb_image(self):
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            cropped = self.raw_image[y:y+h, x:x+w]
            self.offset[0] += x
            self.offset[1] += y
            self.raw_image = cropped
        ratio = self.raw_image.shape[1] / self.raw_image.shape[0]
        if (0.787*(1-0.1) < ratio < 0.787*(1+0.1)):
            h = int(self.raw_image.shape[0] * 0.205)
            self.raw_image = self.raw_image[h:, :]
            self.offset[1] += h
        return self

    def rarity_colors(self):
        def hex2bgr(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

        return list({
            "common": hex2bgr("#7EEF6D"),
            "unusual": hex2bgr("#FFE65D"),
            "rare": hex2bgr("#4d52e3"),
            "epic": hex2bgr("#861FDE"),
            "legendary": hex2bgr("#DE1F1F"),
            "mythic": hex2bgr("#1fdbde"),
            "ultra": hex2bgr("#ff2b75"),
            "super": hex2bgr("#2bffa3")
        }.values())

    def get_mask(self):
        white_mask = cv2.inRange(
            self.image, (255, 255, 255), (255, 255, 255))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            white_mask, connectivity=8)
        if num_labels <= 1:
            return 0, None, None
        max_idx = 1 + np.argmax(stats[1:, 4])
        contour_img = None
        mask = (labels == max_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = self.image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
        self.white_labels = labels
        self.max_white_label = max_idx

    def normalize(self, threshold=40):
        image = self.raw_image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_diff = np.abs(image[:, :, 0] - image[:, :, 1]) + np.abs(
            image[:, :, 1] - image[:, :, 2]) + np.abs(image[:, :, 0] - image[:, :, 2])
        gray_mask = color_diff == 0
        output = np.ones_like(image) * 255
        output[gray_mask & (gray > threshold)] = [255, 255, 255]
        output[gray_mask & (gray <= threshold)] = [0, 0, 0]
        output[~gray_mask] = [255, 255, 255]
        self.image = output
        return self

    def get_end(self):
        white_mask = (self.white_labels == self.max_white_label).astype(
            np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            white_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0, None
        hole_areas = []
        hole_centers = []
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:
                area = cv2.contourArea(contours[i])
                M = cv2.moments(contours[i])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                hole_areas.append(area)
                hole_centers.append((cx, cy))
        if not hole_areas:
            return 0, None
        max_idx = int(np.argmax(hole_areas))
        self.end_p = hole_centers[max_idx]
        return self.end_p

    def get_start(self, color_tol=DEFAULT_COLOR_TOLERANCE):
        max_area = 0
        best_center = None
        max_color = (0, 0, 0)
        for color in self.rarity_colors():
            lower = np.array([max(0, c - color_tol)
                             for c in color], dtype=np.uint8)
            upper = np.array([min(255, c + color_tol)
                             for c in color], dtype=np.uint8)
            mask = cv2.inRange(self.raw_image, lower, upper)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            if num_labels <= 1:
                continue
            max_idx = 1 + np.argmax(stats[1:, 4])
            area = stats[max_idx, 4]
            center = tuple(np.round(centroids[max_idx]).astype(int))
            if area > max_area:
                max_area = area
                best_center = center
                max_color = color

        if best_center is not None:
            self.start_p = best_center
            self.start_color = max_color
            self.inverse_start_color = tuple(
                255 - c for c in max_color)
            return self.start_p
        return None

    def dijkstra(self):
        mask = (self.white_labels == self.max_white_label).astype(
            np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_mask = np.zeros_like(mask)
        cv2.drawContours(area_mask, contours, -1, 255, -1)

        dt = cv2.distanceTransform(area_mask, cv2.DIST_L2, 5)
        h, w = dt.shape
        sx, sy = int(round(self.start_p[0])), int(round(self.start_p[1]))
        ex, ey = int(round(self.end_p[0])), int(round(self.end_p[1]))

        if area_mask[sy, sx] == 0 or area_mask[ey, ex] == 0:
            return None

        dist_map = -np.ones((h, w), dtype=np.float32)
        dist_map[sy, sx] = dt[sy, sx]
        prev = {}
        pq = [(-dt[sy, sx], (sx, sy))]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while pq:
            neg_dist, (x, y) = heappop(pq)
            curr = -neg_dist
            if (x, y) == (ex, ey):
                break
            if curr < dist_map[y, x]:
                continue
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and area_mask[ny, nx] == 255:
                    new_w = min(curr, dt[ny, nx])
                    if new_w > dist_map[ny, nx]:
                        dist_map[ny, nx] = new_w
                        prev[(nx, ny)] = (x, y)
                        heappush(pq, (-new_w, (nx, ny)))

        path = []
        cur = (ex, ey)
        if cur in prev or cur == (sx, sy):
            while cur != (sx, sy):
                path.append(cur)
                cur = prev[cur]
            path.append((sx, sy))
            path.reverse()
            self.sorted_points = path
            return path
        return None

    def rdp(self, epsilon=1):
        if self.sorted_points is None:
            return None
        self.rdp_points = rdp(self.sorted_points, epsilon=epsilon)
        return self.rdp_points

    def extend(self, length=DEFAULT_EXTEND_LENGTH) -> None:
        if self.rdp_points:
            if len(self.rdp_points) < 2:
                self.extend_point = self.rdp_points[0]
                return
            end = self.rdp_points[-1]
            last = self.rdp_points[-2]
            l_l2_dist = distance.euclidean(end, last)
            sine_theta = (end[1] - last[1]) / l_l2_dist
            cosine_theta = (end[0] - last[0]) / l_l2_dist
            self.extend_point = (
                end[0] + length * cosine_theta, end[1] + length * sine_theta)

    def get_final(self, top_left_bound: tuple[int, int] = (0, 0), precise=True) -> list[tuple[int, int]]:
        if self.rdp_points:
            if self.extend_point is not None:
                self.rdp_points.append(self.extend_point)
            ret = [(p[0] + top_left_bound[0], p[1] + top_left_bound[1])
                   for p in self.rdp_points]
        elif self.sorted:
            if self.extend_point is not None:
                self.sorted_points.append(self.extend_point)
            ret = [(p[0] + top_left_bound[0], p[1] + top_left_bound[1])
                   for p in self.sorted_points]
        else:
            ret = []
        if precise:
            return ret
        else:
            return [(int(p[0]), int(p[1])) for p in ret]

    def get_length(self) -> float:
        if self.length:
            return self.length
        if self.rdp_points:
            self.length = sum(distance.euclidean(self.rdp_points[i], self.rdp_points[i + 1])
                              for i in range(len(self.rdp_points) - 1))
        return self.length
    
    def get_width(self):
        if self.width:
            return self.width
        mask = (self.white_labels == self.max_white_label).astype(np.uint8)
        skeleton = skimage_skeletonize(mask > 0).astype(np.uint8)
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        thickness_list = dist_map[skeleton > 0] * 2
        self.width = np.mean(thickness_list)
        return self.width

    def get_difficulty(self, density_r=50) -> float:
        if self.difficulty is not None:
            return self.difficulty
        if self.rdp_points:
            points = np.array(self.rdp_points)
            tree = KDTree(points)
            neighbors = tree.query_ball_point(points, r=density_r)
            counts = np.array([len(n)-1 for n in neighbors])
            density = np.mean(counts)
            lwr = self.get_length()/self.width
            self.difficulty = (max(1, density*3)*lwr *
                               (len(self.rdp_points)**0.5))**0.5
            return self.difficulty
class AFK_Path:
    def __init__(self, raw_points: list[tuple[int, int]], start_p=None, end_p=None, width=None) -> None:
        self.raw_points: list[tuple[int, int]] = raw_points
        self.start_p: tuple[int, int] = start_p
        self.end_p: tuple[int, int] = end_p
        self.rdp_ed: bool = False
        self.rdp_points: list[tuple[int, int]] = None
        self.extend_point: tuple[int, int] = None
        self.sorted: bool = False
        self.sorted_points: list[tuple[int, int]] = None
        self.length: float = None
        self.difficulty: float = None
        self.width: float = width
        self.sort_method = None

    def sort(self) -> None:
        if self.width is None or self.width <= 0:
            self.width = 20
        if self.end_p is not None:
            unused = set(self.raw_points)
            sorted_points = []
            current_point = self.end_p
            while unused:
                next_point = min(
                    unused, key=lambda p: distance.euclidean(current_point, p))
                sorted_points.append(next_point)
                unused.remove(next_point)
                current_point = next_point
            if self.start_p is not None:
                truncated_points = []
                dist = last_dist = float("inf")
                for point in sorted_points:
                    dist = distance.euclidean(point, self.start_p)
                    if not ((dist > last_dist) and (dist < self.width/2)):
                        truncated_points.append(point)
                        last_dist = dist
                    else:
                        truncated_points.append(self.start_p)
                        break
            else:
                truncated_points = sorted_points
            self.sorted_points = truncated_points[::-1]
            self.sorted_points.append(self.end_p)
            self.sort_method = "skeleton"
            self.sorted = True
        else:
            if self.start_p is not None:
                unused = set(self.raw_points)
                sorted_points = []
                current_point = self.start_p
                while unused:
                    next_point = min(
                        unused, key=lambda p: distance.euclidean(current_point, p))
                    sorted_points.append(next_point)
                    unused.remove(next_point)
                    current_point = next_point
                self.sorted_points = sorted_points
                self.sort_method = "skeleton"
                self.sorted = True
            else:
                unused = set(self.raw_points)
                sorted_points = []
                current_point = self.raw_points[0]
                while unused:
                    next_point = min(
                        unused, key=lambda p: distance.euclidean(current_point, p))
                    sorted_points.append(next_point)
                    unused.remove(next_point)
                    current_point = next_point
                self.sorted_points = sorted_points
                self.sort_method = "skeleton"
                self.sorted = True

    def dijkstra(self, mask: torch.Tensor) -> bool:
        if self.start_p is None or self.end_p is None:
            return False
        area_mask = mask.cpu().numpy().astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_mask = np.zeros_like(area_mask)
        cv2.drawContours(area_mask, contours, -1, 255, -1)
        dt = cv2.distanceTransform(area_mask, cv2.DIST_L2, 5)
        h, w = dt.shape
        sx, sy = round(self.start_p[0]), round(self.start_p[1])
        ex, ey = round(self.end_p[0]), round(self.end_p[1])
        if area_mask[sy, sx] == 0 or area_mask[ey, ex] == 0:
            return False
        max_w = -np.ones((h, w), dtype=np.float32)
        prev = {}
        max_w[sy, sx] = dt[sy, sx]
        pq = [(-dt[sy, sx], (sx, sy))]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while pq:
            neg_dist, (x, y) = heappop(pq)
            curr = -neg_dist
            if (x, y) == (ex, ey):
                break
            if curr < max_w[y, x]:
                continue
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if area_mask[ny, nx] == 0:
                    continue
                new_w = min(curr, dt[ny, nx])
                if new_w > max_w[ny, nx]:
                    max_w[ny, nx] = new_w
                    prev[(nx, ny)] = (x, y)
                    heappush(pq, (-new_w, (nx, ny)))
        path = []
        cur = (ex, ey)
        if cur in prev or cur == (sx, sy):
            while cur != (sx, sy):
                path.append(cur)
                cur = prev[cur]
            path.append((sx, sy))
            path.reverse()
        else:
            return False
        self.sorted_points = path
        self.sort_method = "dijkstra"
        self.sorted = True
        return True

    def rdp(self, epsilon=1) -> None:
        if self.sorted:
            if not self.rdp_points:
                self.rdp_points = rdp(
                    self.sorted_points, epsilon=epsilon)
                self.rdp_ed = True
        else:
            if not self.rdp_points:
                self.rdp_points = rdp(self.raw_points, epsilon=epsilon)
                self.rdp_ed = True

    def extend(self, length) -> None:
        if self.rdp_ed:
            if len(self.rdp_points) < 2:
                self.extend_point = self.rdp_points[0]
                return
            end = self.rdp_points[-1]
            last = self.rdp_points[-2]
            l_l2_dist = distance.euclidean(end, last)
            sine_theta = (end[1] - last[1]) / l_l2_dist
            cosine_theta = (end[0] - last[0]) / l_l2_dist
            self.extend_point = (
                end[0] + length * cosine_theta, end[1] + length * sine_theta)
        elif self.sorted:
            end = self.sorted_points[-1]
            last = self.sorted_points[-2]
            l_l2_dist = distance.euclidean(end, last)
            sine_theta = (end[1] - last[1]) / l_l2_dist
            cosine_theta = (end[0] - last[0]) / l_l2_dist
            self.extend_point = (
                end[0] + length * cosine_theta, end[1] + length * sine_theta)
        else:
            raise ValueError(
                "Path is not sorted. Please sort the path before extending.")

    def get_final(self, top_left_bound: tuple[int, int] = (0, 0), precise=True) -> list[tuple[int, int]]:
        if self.rdp_ed:
            if self.extend_point is not None:
                self.rdp_points.append(self.extend_point)
            ret = [(p[0] + top_left_bound[0], p[1] + top_left_bound[1])
                   for p in self.rdp_points]
        elif self.sorted:
            if self.extend_point is not None:
                self.sorted_points.append(self.extend_point)
            ret = [(p[0] + top_left_bound[0], p[1] + top_left_bound[1])
                   for p in self.sorted_points]
        else:
            ret = []
        if precise:
            return ret
        else:
            return [(int(p[0]), int(p[1])) for p in ret]

    def get_length(self) -> float:
        if self.length is not None:
            return self.length
        if self.rdp_ed:
            length = 0
            for i in range(len(self.rdp_points)-1):
                length += distance.euclidean(
                    self.rdp_points[i], self.rdp_points[i+1])
            self.length = length
            return length
        elif self.sorted:
            length = 0
            for i in range(len(self.sorted_points)-1):
                length += distance.euclidean(
                    self.sorted_points[i], self.sorted_points[i+1])
            self.length = length
            return length
        else:
            raise ValueError(
                "Path is not sorted. Please sort the path before getting length.")

    def get_difficulty(self, density_r=50) -> float:
        if self.difficulty is not None:
            return self.difficulty
        if self.rdp_points:
            points = np.array(self.rdp_points)
            tree = KDTree(points)
            neighbors = tree.query_ball_point(points, r=density_r)
            counts = np.array([len(n)-1 for n in neighbors])
            density = np.mean(counts)
            lwr = self.get_length()/self.width
            self.difficulty = (max(1, density*3)*lwr *
                               (len(self.rdp_points)**0.5))**0.5
            return self.difficulty


class AFK_Segment:
    def __init__(self, afk_window_image: cv2.Mat, mask: torch.Tensor, start: tuple[int, int], end: tuple[int, int], start_size: int) -> None:
        self.image = afk_window_image
        self.mask = mask
        self.start = start
        self.end = end
        self.width = None
        self.start_size = start_size
        self.segmented_path = None
        if start is not None:
            self.start_color = tuple(int(i) for i in tuple(
                afk_window_image[int(start[1]), int(start[0])]))
            self.inverse_start_color = tuple(int(i) for i in tuple(
                255 - afk_window_image[int(start[1]), int(start[0])]))
        else:
            self.start_color = (0, 0, 0)
            self.inverse_start_color = (255, 255, 255)

    def save_start(self) -> None:
        if self.mask.ndim == 2:
            H, W = self.mask.shape
        elif self.mask.ndim == 3:
            C, H, W = self.mask.shape
        else:
            pass
        device = self.mask.device
        dtype = self.mask.dtype
        radius = self.start_size / 2.0
        radius_sq = radius ** 2
        cx, cy = int(self.start[0]), int(self.start[1])
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device),
                                            indexing='ij')
        dist_sq = (x_coords.float() - cx)**2 + (y_coords.float() - cy)**2
        circle_mask_bool = dist_sq <= radius_sq
        circle_mask = circle_mask_bool.to(dtype)
        if self.mask.ndim == 3:
            circle_mask = circle_mask.unsqueeze(0)
        self.mask = torch.maximum(self.mask, circle_mask)

    def segment_path(self) -> list[tuple[int, int]]:
        mask = self.mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        skeleton = cv2.ximgproc.thinning(mask)
        contours, _ = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        line = [(point[0][0], point[0][1]) for point in contour]
        line = list(set(line))
        self.segmented_path = line
        return line

    def get_width(self):
        if self.width is None:
            mask_np = self.mask.cpu().numpy()
            mask_bin = (mask_np > 0).astype(np.uint8)
            skeleton = skimage_skeletonize(mask_bin).astype(np.uint8)
            dist_map = distance_transform_edt(mask_bin)
            thickness_list = dist_map[skeleton > 0] * 2
            thickness_list = dist_map[skeleton > 0] * 2
            self.width = np.mean(thickness_list)
        return self.width



def apply_mouse_movement(points, speed=100, active=False):
    pyautogui.moveTo(points[0][0], points[0][1], duration=0.5)
    pyautogui.mouseUp(button="left")
    pyautogui.mouseUp(button="right")
    if active:
        pyautogui.doubleClick()
    pyautogui.mouseDown(button="left")
    for i in range(1, len(points), 1):
        distance_ = distance.euclidean(points[i], points[i-1])
        if distance_ < 5:
            pyautogui.moveTo(points[i][0], points[i][1])
        else:
            duration = distance_ / speed
            pyautogui.moveTo(points[i][0], points[i][1], duration=duration)
    pyautogui.mouseUp(button="left")



def execute_afk_bw(afk_window_pos, image,speed=100, 
                color_tolerance=DEFAULT_COLOR_TOLERANCE, 
                rdp_epsilon=DEFAULT_RDP_EPSILON, 
                extend_length=DEFAULT_EXTEND_LENGTH):
    print("Using BW OpenCV")
    afk_bw = AFK_BW(image)
    afk_bw.crop_nb_image().normalize().get_mask()
    start_p, end_p = afk_bw.get_start(color_tol=color_tolerance), afk_bw.get_end()
    if start_p is None :
        print("No start point found")
    elif end_p is None:
        print("No end point found")
    else:
        afk_bw.dijkstra()
        afk_bw.rdp(round(eval(rdp_epsilon.replace(
            "width", str(afk_bw.get_width())))))
        afk_bw.extend(extend_length)
        line = afk_bw.get_final(afk_window_pos[0], precise=False)
        start_p = (start_p[0] + afk_bw.offset[0],
                   start_p[1] + afk_bw.offset[1])
        end_p = (end_p[0] + afk_bw.offset[0],
                 end_p[1] + afk_bw.offset[1])
        line = [(p[0] + afk_bw.offset[0], p[1] + afk_bw.offset[1])
                for p in line]
    ori_pos = pyautogui.position()
    apply_mouse_movement(line, speed=speed, active=True)
    pyautogui.moveTo(ori_pos[0], ori_pos[1], duration=0.1)

def execute_afk_segment(afk_window_pos, image, afk_seg_model=afk_seg_model, afk_det_model=afk_det_model,speed=100,rdp_epsilon=DEFAULT_RDP_EPSILON, extend_length=DEFAULT_EXTEND_LENGTH):
    res = get_masks_by_iou(image, afk_seg_model)
    start_p, end_p, start_size, pack = detect_afk_things(
        image, afk_det_model, caller="main")
    if res is None:
        print("No masks found", "ERROR")
    position = start_p, end_p, afk_window_pos, start_size
    mask, results = res
    if start_p is None:
        print("No start point found, going for AUTO prediction")
    if end_p is None:
        print("No end found, going for LINEAR prediction")
    
    # Move this line outside the conditional block
    afk_mask = AFK_Segment(image, mask, start_p, end_p, start_size)
    
    if start_p is not None:
        afk_mask.save_start()    
    afk_path = AFK_Path(afk_mask.segment_path(), start_p,
                        end_p, afk_mask.get_width())
    dijkstra_stat = afk_path.dijkstra(afk_mask.mask)
    if not dijkstra_stat:
        afk_path.sort()
    afk_path.rdp(round(eval(rdp_epsilon.replace(
        "width", str(afk_mask.get_width())))))
    afk_path.extend(extend_length)
    line = afk_path.get_final(afk_window_pos[0], precise=False)
    ori_pos = pyautogui.position()
    print("Using Segmentation Model")
    apply_mouse_movement(line, speed=speed, active=True)
    pyautogui.moveTo(ori_pos[0], ori_pos[1], duration=0.1)


def random_key_press(key_press_delay=DEFAULT_KEY_PRESS_DELAY):
    keys = ['a', 's', 'w', 'd']
    random_keynum = random.randint(0, 3)
    pyautogui.keyDown(keys[random_keynum])
    time.sleep(key_press_delay)
    pyautogui.keyUp(keys[random_keynum])
    pyautogui.keyDown(keys[3-random_keynum])
    time.sleep(key_press_delay)
    pyautogui.keyUp(keys[3-random_keynum])

def crop_image(left_top_bound, right_bottom_bound, image):
    return image[left_top_bound[1]:right_bottom_bound[1], left_top_bound[0]:right_bottom_bound[0]]

def get_config() -> dict:
    with open("./config.json", "r", encoding="utf-8") as f:
        return json.load(f)
