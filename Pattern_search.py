import cv2
import numpy as np

# ------------------------ Configuration ------------------------
# Define constants and parameters to make tuning easy
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
LOWE_RATIO = 0.6                # Ratio test threshold for good SIFT matches
MIN_GOOD_MATCHES = 10           # Minimum good matches to consider a valid detection
CANNY_THRESHOLD1 = 50           # First threshold for Canny edge detection
CANNY_THRESHOLD2 = 255          # Second threshold for Canny edge detection
KERNEL = np.ones((3, 3), np.uint8)  # Kernel used for dilating edge image
GREEN_IMG = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (0, 255, 0), np.uint8)  # Green overlay image

# ------------------------ Global Variables ------------------------
# Track region-of-interest selection and tracking status
roi_start = ()
roi_end = ()
selected_roi_start = ()
selected_roi_end = ()
selection_complete = False      # True after ROI is selected
tracking_enabled = False        # True once reference image is ready for matching
is_drawing = False              # Mouse is actively drawing ROI

# ------------------------ Setup ------------------------
# Initialize video capture and SIFT feature detector
cap = cv2.VideoCapture(1)
sift = cv2.SIFT_create()
index_params = dict(algorithm=0, trees=5)
search_params = dict()
cv2.namedWindow("Frame")

# ------------------------ Mouse Callback ------------------------
def mouse_drawing(event, x, y, flags, params):
    # Handles drawing of the region of interest (ROI) with mouse clicks
    global roi_start, roi_end, selected_roi_start, selected_roi_end, is_drawing, selection_complete
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_drawing:
            is_drawing = True
            roi_start = (x, y)
        else:
            is_drawing = False
            selected_roi_start = roi_start
            selected_roi_end = roi_end
            selection_complete = True
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        roi_end = (x, y)

cv2.setMouseCallback("Frame", mouse_drawing)

# ------------------------ Utility Functions ------------------------
def compute_similarity(ref_edges, detected_edges):
    # Calculates how similar the detected object is to the reference based on edge overlap
    match_pixels = np.sum(cv2.bitwise_and(ref_edges, detected_edges) == 255)
    total_pixels = np.sum(ref_edges == 255)
    return (match_pixels / total_pixels) * 100 if total_pixels > 0 else 0

def draw_rectangle(frame):
    # Visual feedback while drawing ROI
    if roi_start and roi_end and is_drawing:
        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 1)

# ------------------------ Main Processing ------------------------
flann = None
reference_img = None
reference_edges = None
kp_ref = None
desc_ref = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    draw_rectangle(frame)

    # Step 1: Extract reference image after ROI is selected
    if selection_complete:
        x1, y1 = selected_roi_start
        x2, y2 = selected_roi_end
        if x2 > x1 and y2 > y1:
            reference_img = frame[y1:y2, x1:x2]
            reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            reference_edges = cv2.Canny(reference_gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
            kp_ref, desc_ref = sift.detectAndCompute(reference_gray, None)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            tracking_enabled = True
        selection_complete = False

    # Step 2: Track the reference in live video using SIFT and homography
    if tracking_enabled and desc_ref is not None:
        kp_frame, desc_frame = sift.detectAndCompute(frame_gray, None)
        if desc_frame is None or len(kp_frame) < 2:
            continue

        # Apply Lowe's ratio test to filter good matches
        matches = flann.knnMatch(desc_ref, desc_frame, k=2)
        good_matches = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

        if len(good_matches) > MIN_GOOD_MATCHES:
            # Extract matching keypoint coordinates
            query_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography and transform object corners
            matrix, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            if matrix is None:
                continue

            matrix_inv = np.linalg.pinv(matrix)
            h, w = reference_img.shape[:2]
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            # Warp frame to reference perspective and compute edges
            warp_img = cv2.warpPerspective(frame, matrix_inv, (w, h))
            warp_edges = cv2.Canny(warp_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
            warp_dilated = cv2.dilate(warp_edges, KERNEL, iterations=1)

            # Calculate edge-based match mask
            match_mask = cv2.bitwise_and(reference_edges, warp_dilated)
            mask = cv2.warpPerspective(match_mask, matrix, (FRAME_WIDTH, FRAME_HEIGHT))
            mask_inv = cv2.bitwise_not(mask)

            # Create a color visualization of the match
            matched_area = cv2.bitwise_and(frame, frame, mask=mask)
            non_matched_area = cv2.bitwise_and(frame, frame, mask=mask_inv)
            difference = cv2.bitwise_and(GREEN_IMG, matched_area)
            result = cv2.add(non_matched_area, difference)

            # Display similarity score as overlay
            similarity = compute_similarity(reference_edges, warp_dilated)
            font = cv2.FONT_ITALIC
            cv2.putText(result, f"Match = {int(similarity)}%", (400, 30), font, 1, (0, 255, 0), 2)

            # Draw bounding box around matched object in live frame
            frame = cv2.polylines(result, [np.int32(dst)], True, (255, 0, 0), 2)

            # Debug output for inspection
            cv2.imshow("Reference", reference_img)
            cv2.imshow("Warped", warp_img)
            cv2.imshow("Edges", warp_edges)
            cv2.imshow("Match Mask", match_mask)

    # Always show the live feed
    cv2.imshow("Frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
