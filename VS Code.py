import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

print("=" * 60)
print("TASK 4: OBJECT DETECTION AND TRACKING")
print("CodeAlpha AI Internship")
print("=" * 60)

# Load YOLOv8 model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("✅ Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

# Get default webcam resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Webcam connected! Resolution: {frame_width}x{frame_height}")

# Create window and set to fullscreen
window_name = 'Task 4: Object Detection + Tracking - CodeAlpha'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n📌 This system detects objects and assigns TRACKING IDs")
print("\n🎮 Controls:")
print("   Press 'q' - Quit")
print("   Press 's' - Save screenshot")
print("   Press 't' - Toggle tracking info")
print("   Press 'f' - Toggle fullscreen")
print("   Press 'r' - Reset tracking history")
print("=" * 60)

# Store previous frame's tracking data
track_history = defaultdict(lambda: [])
track_colors = {}

# Variables
fullscreen = True
show_tracking_info = True
frame_count = 0

# Function to generate consistent colors for each track ID
def get_color(track_id):
    if track_id not in track_colors:
        # Generate a unique color for each tracking ID
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        track_colors[track_id] = color
    return track_colors[track_id]

# Function to toggle fullscreen
def toggle_fullscreen():
    global fullscreen
    fullscreen = not fullscreen
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print("📺 Fullscreen mode: ON")
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        print("📺 Fullscreen mode: OFF")

while True:
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error reading from webcam")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    frame_count += 1
    
    # Get frame dimensions for responsive UI
    h, w = frame.shape[:2]
    
    # Run YOLO detection and tracking
    results = model.track(frame, persist=True, verbose=False)
    
    # Get tracking IDs and bounding boxes
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Draw detections and tracking
        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(class_id)]
            
            # Only show detections with >50% confidence
            if conf > 0.5:
                # Get color for this track ID
                color = get_color(int(track_id))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Create labels
                label = f"{class_name}: {conf:.2f}"
                tracking_label = f"ID: {int(track_id)}"
                
                # Calculate text size for responsive positioning
                font_scale = 0.6 if w > 640 else 0.4
                thickness = 2 if w > 640 else 1
                
                # Draw label background for object name
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                # Draw tracking ID
                cv2.putText(frame, tracking_label, (x1, y2 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                
                # Draw center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
                
                # Store track history
                track_history[int(track_id)].append((center_x, center_y))
                
                # Draw tracking trail (last 20 positions)
                if len(track_history[int(track_id)]) > 1:
                    points = track_history[int(track_id)][-30:]
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], color, 2)
    
    # Create responsive information panel (top-left)
    panel_width = 400 if w > 800 else 300
    panel_height = 240 if h > 600 else 200
    cv2.rectangle(frame, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (panel_width, panel_height), (0, 255, 0), 2)
    
    # Panel content with responsive font sizes
    title_font = 0.5 if w > 800 else 0.4
    normal_font = 0.4 if w > 800 else 0.35
    small_font = 0.35 if w > 800 else 0.3
    
    y_offset = 25
    cv2.putText(frame, "CODEALPHA - TASK 4", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, title_font, (0, 255, 0), 1)
    y_offset += 22
    
    cv2.putText(frame, "Object Detection + Tracking", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, small_font, (255, 255, 255), 1)
    y_offset += 18
    
    cv2.putText(frame, f"Model: YOLOv8 | Frame: {frame_count}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, small_font, (200, 200, 200), 1)
    y_offset += 22
    
    # Count active tracked objects
    active_tracks = len(track_history)
    cv2.putText(frame, f"Active Tracks: {active_tracks}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, normal_font, (0, 255, 255), 1)
    y_offset += 22
    
    # Show tracking status
    if show_tracking_info:
        cv2.putText(frame, "TRACKING: ACTIVE", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 0), 1)
        y_offset += 18
        cv2.putText(frame, "Each object has unique ID", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, small_font - 0.05, (200, 200, 200), 1)
        y_offset += 16
        cv2.putText(frame, "Trail shows movement path", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, small_font - 0.05, (200, 200, 200), 1)
        y_offset += 16
    else:
        cv2.putText(frame, "TRACKING INFO: HIDDEN", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, small_font, (100, 100, 100), 1)
        y_offset += 18
    
    # Controls panel (bottom-left)
    controls_y = h - 60
    cv2.rectangle(frame, (0, controls_y - 5), (panel_width, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, controls_y - 5), (panel_width, h), (0, 255, 0), 1)
    
    cv2.putText(frame, "CONTROLS:", (10, controls_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, small_font, (0, 255, 255), 1)
    cv2.putText(frame, "Q:Quit  S:Screen  T:Info  F:Fullscreen  R:Reset", (10, controls_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, small_font - 0.05, (200, 200, 200), 1)
    
    # Live indicator (top-right)
    cv2.circle(frame, (w - 25, 30), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (w - 60, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Fullscreen indicator
    if fullscreen:
        cv2.putText(frame, "FULLSCREEN", (w - 120, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Display the frame
    cv2.imshow(window_name, frame)
    
    # Key controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n🛑 Stopping detection...")
        break
    
    elif key == ord('s'):
        filename = f"detection_tracking_{frame_count}.png"
        cv2.imwrite(filename, frame)
        print(f"\n📸 Screenshot saved: {filename}")
        print(f"   Active objects being tracked: {active_tracks}")
        print("=" * 50)
    
    elif key == ord('t'):
        show_tracking_info = not show_tracking_info
        print(f"\n📋 Tracking info: {'ON' if show_tracking_info else 'OFF'}")
    
    elif key == ord('f'):
        toggle_fullscreen()
    
    elif key == ord('r'):
        track_history.clear()
        track_colors.clear()
        print("\n🔄 Tracking history reset!")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("✅ TASK 4 COMPLETED SUCCESSFULLY")
print("   - Object Detection: YOLOv8")
print("   - Object Tracking: Built-in tracker")
print("   - Tracking IDs assigned to each object")
print("   - Fullscreen mode available")
print("=" * 60)
