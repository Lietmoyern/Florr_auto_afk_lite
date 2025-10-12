from ultralytics import YOLO
import numpy as np



model_path = "./models/afk-det.pt"
afk_det_model = YOLO(model_path)
seg_model_path = "./models/afk-seg.pt"
afk_seg_model = YOLO(seg_model_path)



DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_WINDOW_RATIO_TOLERANCE = 0.1
DEFAULT_WINDOW_ASPECT_RATIOS = [1.0, 0.787]

def yolo_detect(model, img, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    result = model.predict(img, verbose=False)[0]
    names = result.names
    boxes = result.boxes
    things = boxes.data.tolist()
    detected = []
    
    for method in things:
        new_method = []
        for i in method:
            new_method.append(round(i))
        
        x_1 = new_method[0]
        y_1 = new_method[1]
        x_2 = new_method[2]
        y_2 = new_method[3]
        confidence = new_method[4]  
        name = new_method[5]
        object_name = names[name]
        

        if confidence >= confidence_threshold:
            detected.append({
                "name": object_name,
                "x_1": x_1,
                "y_1": y_1,
                "x_2": x_2,
                "y_2": y_2,
                "x_avg": (x_1 + x_2) / 2,
                "y_avg": (y_1 + y_2) / 2,
                "confidence": confidence
            })
    
    return detected, img

def detect_afk_window(img, afk_det_model, 
                      confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, 
                      window_ratio_tolerance=DEFAULT_WINDOW_RATIO_TOLERANCE, 
                      window_aspect_ratios=DEFAULT_WINDOW_ASPECT_RATIOS):
    things, _ = yolo_detect(afk_det_model, img, confidence_threshold)
    

    window_candidates = []
    for thing in things:
        if thing['name'] == 'Window':
            window_candidates.append(thing)
    

    if not window_candidates:
        return None
    

    window_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    best_window = window_candidates[0]
    

    windows_pos = ((best_window['x_1'], best_window['y_1']),
                   (best_window['x_2'], best_window['y_2']))
    
    window_width = windows_pos[1][0] - windows_pos[0][0]
    window_height = windows_pos[1][1] - windows_pos[0][1]
    
    if window_height == 0:
        return None
    
    ratio = window_width / window_height
    

    ratio_matched = False
    for aspect_ratio in window_aspect_ratios:
        if (aspect_ratio * (1 - window_ratio_tolerance) < ratio < aspect_ratio * (1 + window_ratio_tolerance)):
            ratio_matched = True
            break
    
    if ratio_matched:
        return windows_pos
    
    return None

def detect_afk_things(cropped_img, afk_det_model, caller="main", test_time=None):
    afk_window_img = cropped_img.copy()
    things_afk = yolo_detect(afk_det_model, afk_window_img)
    start_pos = end_pos = None
    start_max_confidence = end_max_confidence = 0
    start_size = 0
    pack = [None, None]
    for thing in things_afk[0]:
        if thing['name'] == 'Start' and thing['confidence'] > start_max_confidence:
            start_pos = (thing['x_avg'], thing['y_avg'])
            start_max_confidence = thing['confidence']
            start_size = (abs(thing['x_2'] - thing['x_1']) +
                          abs(thing['y_2'] - thing['y_1'])) / 2
            pack[0] = [(thing['x_1'], thing['y_1']),
                       (thing['x_2'], thing['y_2'])]
        if thing['name'] == 'End' and thing['confidence'] > end_max_confidence:
            end_pos = (thing['x_avg'], thing['y_avg'])
            end_max_confidence = thing['confidence']
            pack[1] = [(thing['x_1'], thing['y_1']),
                       (thing['x_2'], thing['y_2'])]
    return start_pos, end_pos, start_size, pack

def get_masks_by_iou(image, afk_seg_model: YOLO, lower_iou=0.3, upper_iou=0.7, stepping=0.1):
    for iou in np.arange(upper_iou, lower_iou, -stepping):
        results = afk_seg_model.predict(
            image, retina_masks=True, verbose=False, iou=iou)
        if results[0].masks is None:
            return None
        if len(results[0].masks.data) == 1:
            break
    if len(results[0].masks.data) != 1:
        results.sort(key=lambda x: x.boxes.conf[0], reverse=True)
    mask = results[0].masks.data[0]
    return mask, results