import cv2
import os
import glob
import time

from datetime import datetime
from ultralytics import YOLO


candidates = glob.glob("runs/**/crop_health_classifier_v2/weights/best.pt", recursive=True)
if not candidates:
    raise FileNotFoundError("No best.pt found for crop_health_classifier_v2.")
model_path = candidates[0]
print(f"✅ Using model: {model_path}")

model = YOLO(model_path)

# --- Step 3: Open Camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected!")
    exit()

os.makedirs("detections", exist_ok=True)
print("🌿 Camera ON — press 'q' to quit\n")

last_label = "HEALTHY"
last_save_time = 0
cooldown = 5  # seconds between actions

# --- Step 4: Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    x1, y1, x2, y2 = int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    results = model.predict(source=roi, save=False, verbose=False)
    r = results[0]
    names = r.names  # {0:'dead', 1:'healthy'}
    probs = r.probs.data.tolist()
    dead_conf, healthy_conf = probs[0]*100, probs[1]*100

    print(f"HEALTHY: {healthy_conf:.1f}% | DEAD: {dead_conf:.1f}%")

    # --- Step 5: Decision Logic ---
    if healthy_conf - dead_conf > 10:
        final_label, color = "HEALTHY", (0, 255, 0)
    elif dead_conf - healthy_conf > 10:
        final_label, color = "DEAD", (0, 0, 255)
    else:
        final_label, color = "UNCERTAIN", (0, 255, 255)

    # --- Step 6: Trigger Arduino when Dead detected ---
    now = time.time()
    if final_label == "DEAD" and (last_label != "DEAD" or now - last_save_time > cooldown):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detections/dead_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"🟥 Dead plant detected — saved {filename}")

       
        

        last_save_time, last_label = now, "DEAD"

    elif final_label != "DEAD":
        last_label = final_label

    # --- Step 7: Display Output ---
    cv2.putText(frame, final_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    cv2.imshow("🌾 Crop Health Live Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Step 8: Cleanup ---
cap.release()
cv2.destroyAllWindows()

print("🌿 Camera OFF — goodbye!")
