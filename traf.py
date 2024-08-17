import cv2
import numpy as np
import threading

# Parameters for vehicle detection
min_contour_width = 40
min_contour_height = 40
line_height = 505
upper_line_height = 500
vehicles_lane1 = 0
vehicles_lane2 = 0
center_x = 0

lock = threading.Lock()

def get_centroid(x, y, w, h):
    """Calculate the centroid of a bounding box."""
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

def is_within_lane(cx, lane):
    """Check if the centroid is within the specified lane."""
    global center_x
    if lane == 1:
        return cx < center_x  # Assuming lane1 is to the left of center
    elif lane == 2:
        return cx >= center_x  # Assuming lane2 is to the right of center
    return False

def process_frame(frame1, frame2, bg_subtractor):
    global vehicles_lane1, vehicles_lane2, center_x

    # Compute the difference between frames
    fg_mask = bg_subtractor.apply(frame2)

    # Apply thresholding
    _, th = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((5, 5)), iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frame_height, frame_width = frame1.shape[:2]
    center_x = frame_width // 2

    # Draw lane dividers
    cv2.line(frame1, (center_x, 0), (center_x, frame_height), (0, 255, 0), 2)  # Vertical line for lane separation
    cv2.line(frame1, (0, line_height), (frame_width, line_height), (0, 255, 0), 2)
    cv2.line(frame1, (0, upper_line_height), (frame_width, upper_line_height), (0, 255, 0), 2)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)  # Extract bounding box
        if w >= min_contour_width and h >= min_contour_height:
            centroid = get_centroid(x, y, w, h)
            cx, cy = centroid
            if upper_line_height < cy < line_height:
                with lock:
                    if is_within_lane(cx, 1):
                        vehicles_lane1 += 1
                    elif is_within_lane(cx, 2):
                        vehicles_lane2 += 1

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the vehicle count for each lane
    cv2.putText(frame1, "Lane 1 Vehicles: " + str(vehicles_lane1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)
    cv2.putText(frame1, "Lane 2 Vehicles: " + str(vehicles_lane2), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 2)

    return frame1

def process_video():
    cap = cv2.VideoCapture('vi.mp4', cv2.CAP_FFMPEG)
    cap.set(3, 1920)
    cap.set(4, 1080)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Vehicle Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while ret:
        processed_frame = process_frame(frame1, frame2, bg_subtractor)
        cv2.imshow("Vehicle Detection", processed_frame)

        if cv2.waitKey(1) == 27:
            break

        frame1 = frame2
        ret, frame2 = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    threading.Thread(target=process_video).start()
