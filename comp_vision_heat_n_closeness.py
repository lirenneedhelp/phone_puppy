import cv2
import numpy as np
import torch

# 1) Load MiDaS (small = faster)
model_type = "MiDaS_small"  # fast
midas = torch.hub.load("intel-isl/MiDaS", model_type)  # downloads model first time
midas.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
midas.to(device)

# 2) Load transforms (preprocessing)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # for MiDaS_small

# 3) Open webcam
cap = cv2.VideoCapture("http://192.168.137.24:81/stream")
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to (1).")

# If this feels too sensitive, increase it a bit.
CLOSE_THRESHOLD = 0.70  # 0..1, higher = must be VERY close to trigger STOP

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare input for MiDaS
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize prediction to original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize to 0..1 (relative depth)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # MiDaS outputs "inverse depth" style relative maps; close areas tend to be brighter after normalization.
    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # Check "closeness" in the center region
    h, w = depth_norm.shape
    cx1, cx2 = int(w * 0.4), int(w * 0.6)
    cy1, cy2 = int(h * 0.4), int(h * 0.6)
    center_patch = depth_norm[cy1:cy2, cx1:cx2]
    center_value = float(center_patch.mean())

    status = "GO"
    if center_value > CLOSE_THRESHOLD:
        status = "STOP"

    # Draw box + status
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
    cv2.putText(frame, f"{status}  closeness={center_value:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if status=="STOP" else (0,255,0), 2)

    # Show two windows
    cv2.imshow("Webcam", frame)
    cv2.imshow("MiDaS Depth (near=bright)", depth_vis)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):  # ESC or q to quit
        break

cap.release()
cv2.destroyAllWindows()
