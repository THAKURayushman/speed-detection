import cv2
import time

# Load cascade classifier
car_cascade = cv2.CascadeClassifier("haarcascade_car.xml")

# Load video
cap = cv2.VideoCapture("traffic2.mp4")
if not cap.isOpened():
    print("❌ Could not open video")
    exit()

# Constants
LINE_Y = 200  # y-position of the virtual line
DISTANCE_METERS = 10  # Assumed real-world distance vehicles travel (in meters)

# Storage
vehicle_id = 0
vehicle_tracker = {}          # bounding box -> timestamp
vehicle_cross_time = {}       # id -> timestamp
vehicle_speed = {}            # id -> speed

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    # Draw line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)

    # Detect cars and check if they cross the line
    for (x, y, w, h) in cars:
        cx = x + w // 2
        cy = y + h // 2

        if LINE_Y - 5 < cy < LINE_Y + 5:
            bbox = (x, y, w, h)
            if bbox not in vehicle_tracker:
                vehicle_tracker[bbox] = time.time()
                vehicle_id += 1
                vehicle_cross_time[vehicle_id] = vehicle_tracker[bbox]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate speed (once per vehicle)
    for vid, t1 in list(vehicle_cross_time.items()):
        if vid not in vehicle_speed:
            elapsed = time.time() - t1
            if elapsed > 0:
                speed = DISTANCE_METERS / elapsed * 3.6  # m/s to km/h
                vehicle_speed[vid] = speed
                print(f"✅ Vehicle {vid} speed: {speed:.2f} km/h")
            else:
                print(f"⚠️ Skipped speed calculation for vehicle {vid} (elapsed time was zero)")

    # Overlay speeds on frame
    for i, (vid, speed) in enumerate(vehicle_speed.items()):
        text = f"ID:{vid} Speed:{speed:.2f} km/h"
        cv2.putText(frame, text, (50, 50 + 20 * i), font, 0.6, (255, 255, 255), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
