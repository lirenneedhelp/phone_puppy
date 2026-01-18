import cv2
import numpy as np
import socket
import threading

# --- CONFIGURATION ---
ESP32_IP = "192.168.137.242"
STREAM_URL = f"http://{ESP32_IP}:81/stream"
UDP_PORT = 4210

CONFIRM_ON_FRAMES = 3   # Faster response
CONFIRM_OFF_FRAMES = 15 # Wait ~1 second of emptiness before turning off

# Create UDP socket once
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

class VideoStream:
    """Helper class to read frames in a separate thread to eliminate lag."""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

def send_signal_udp(state):
    try:
        message = b"1" if state else b"0"
        udp_socket.sendto(message, (ESP32_IP, UDP_PORT))
        # print(f"UDP Sent: {state}")
    except Exception as e:
        print(f"UDP Error: {e}")

def run_detection():
    # Load YOLO model
    net = cv2.dnn.readNet("yolo_params/yolov3-tiny.weights", "yolo_params/yolov3-tiny.cfg")
    # Try to use GPU if available
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Start threaded stream
    stream = VideoStream(STREAM_URL)
    
    is_person_present = False
    on_counter = 0
    off_counter = 0

    print("System started (UDP Mode). Press 'q' to exit.")

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            continue

        # (160, 160) is very fast, but if detection is poor, increase to (320, 320)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (160, 160), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        detected_this_frame = False
        
        # Optimized Vectorized Detection
        for out in outs:
            scores = out[:, 5:]
            class_ids = np.argmax(scores, axis=1)
            confidences = scores[np.arange(len(scores)), class_ids]
            
            # Mask for "person" (index 0) with high confidence
            mask = (confidences > 0.5) & (class_ids == 0)
            if np.any(mask):
                detected_this_frame = True
                break

        # --- HYSTERESIS LOGIC ---
        if detected_this_frame:
            on_counter += 1
            off_counter = 0
        else:
            off_counter += 1
            on_counter = 0

        # Change state to ON
        if not is_person_present and on_counter >= CONFIRM_ON_FRAMES:
            is_person_present = True
            send_signal_udp(1)
            print("🔔 PERSON DETECTED")

        # Change state to OFF
        elif is_person_present and off_counter >= CONFIRM_OFF_FRAMES:
            is_person_present = False
            send_signal_udp(0)
            print("👋 AREA CLEAR")
        
        cv2.imshow("UDP Detection Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()
    udp_socket.close()