import asyncio
import cv2
import numpy as np
import pyautogui
import time
import threading
from detector import detect_afk_window,afk_det_model,afk_seg_model
from helpful_functions import execute_afk_bw, execute_afk_segment, random_key_press, crop_image, get_config
from connection import main as websocket_main, set_state, connected_clients


config = get_config()

pyautogui.FAILSAFE = config['mouse']['failsafe']


running = True
AFK_DETECTION_INTERVAL = config['afk_detection']['interval']
timing_config = config['timing']


afk_config = config['afk_detection']
mouse_config = config['mouse']
keyboard_config = config['keyboard']
image_processing_config = config['image_processing']


confidence_threshold = afk_config['confidence_threshold']
window_ratio_tolerance = afk_config['window_ratio_tolerance']
window_aspect_ratios = afk_config['window_aspect_ratios']


color_tolerance = image_processing_config['color_tolerance']
rdp_epsilon = image_processing_config['rdp_epsilon']
extend_length = image_processing_config['extend_length']
key_press_delay = keyboard_config['key_press_delay']
mouse_speed = mouse_config['speed']


auto_exit_minutes = config['timing']['auto_exit_minutes']
method=config['method']['use_afk_bw']
# True for afk_bw, False for afk_segment,use websocket only when method is True
if auto_exit_minutes > 0:
    def auto_exit_timer():
        global running
        print(f"Running for {auto_exit_minutes} minutes")
        time.sleep(auto_exit_minutes * 60)  
        print("Time's up, auto exiting...")
        running = False
    
    exit_thread = threading.Thread(target=auto_exit_timer, daemon=True)
    exit_thread.start()
else:
    print("Running indefinitely")

    
def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

global_loop = asyncio.new_event_loop()
asyncio.set_event_loop(global_loop)


def process_afk_detection():
    global running
    
    print("AFK Detector Lite started...")
    print("A lite version of https://github.com/Shiny-Ladybug/florr-auto-afk")
    print("Use full version for advanced features if it works for you.")
    while running:
        try:
            frame = capture_screen()
            
            afk_window_pos = detect_afk_window(
                frame, 
                afk_det_model, 
                confidence_threshold=confidence_threshold,
                window_ratio_tolerance=window_ratio_tolerance,
                window_aspect_ratios=window_aspect_ratios
            )
            
            if afk_window_pos is not None:
                print(f"AFK window found at: {afk_window_pos}")
                
                global global_loop
                if global_loop.is_closed():
                    global_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(global_loop)
                global_loop.run_until_complete(set_state(True))
                

                time.sleep(timing_config['page_opacity_wait'])
                ori_frame = frame.copy()
                frame = capture_screen()  
                left_top_bound = afk_window_pos[0]
                right_bottom_bound = afk_window_pos[1]
                afk_window_img = crop_image(left_top_bound, right_bottom_bound, frame)
                spare_afk_window_img = crop_image(left_top_bound, right_bottom_bound, ori_frame)
                cv2.imwrite(f"./logs/afk_window_{time.time()}.png", afk_window_img) 
                
                if(method==True and len(connected_clients) > 0):
                    try:
                        execute_afk_bw(
                            afk_window_pos, 
                            afk_window_img,
                            speed=mouse_speed,
                            color_tolerance=color_tolerance,
                            rdp_epsilon=rdp_epsilon,
                            extend_length=extend_length
                        )
                    except Exception as e:
                        print(f"execute_afk_bw failed: {e}, trying execute_afk_segment with spare image")
                        execute_afk_segment(
                            afk_window_pos, 
                            spare_afk_window_img,
                            speed=mouse_speed,
                            rdp_epsilon=rdp_epsilon,
                            extend_length=extend_length
                        )
                else:
                    execute_afk_segment(
                        afk_window_pos, 
                        afk_window_img,
                        speed=mouse_speed,
                        rdp_epsilon=rdp_epsilon,
                        extend_length=extend_length
                    )

                random_key_press(key_press_delay=key_press_delay)
                

                time.sleep(timing_config['post_action_wait'])
                

                if global_loop.is_closed():
                    global_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(global_loop)
                global_loop.run_until_complete(set_state(False))
                
                print("AFk window handled, waiting for next detection...")
            

            time.sleep(AFK_DETECTION_INTERVAL)
            
        except Exception as e:
            print(f"AFK Detector Lite error: {e}")

            try:
                if global_loop.is_closed():
                    global_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(global_loop)
                global_loop.run_until_complete(set_state(False))
            except:
                pass
            time.sleep(AFK_DETECTION_INTERVAL)

def start_websocket_server():
    try:
        asyncio.run(websocket_main())
    except KeyboardInterrupt:
        print("WebSocket Server stopped")
    except Exception as e:
        print(f"WebSocket Server error: {e}")

def main():
    global running
    
    print("Florr Auto AFK Detector Lite")
    

    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    if(method==True):
        ws_thread.start()
    

    time.sleep(1)
    

    detection_thread = threading.Thread(target=process_afk_detection, daemon=True)
    detection_thread.start()
    

    while running:
        time.sleep(1)
    

    print("Program stopping...")

    if ws_thread.is_alive():
        ws_thread.join(1)  
    if detection_thread.is_alive():
        detection_thread.join(1)  
    print("Program stopped")

if __name__ == "__main__":
    main()