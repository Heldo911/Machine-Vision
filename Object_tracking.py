import  cv2
import  numpy as np

point_1 = ()
point_2 = ()
flag = False
enable = False
drawing = False

cap = cv2.VideoCapture(1)

def mouse_drawing(event,x,y,flags,params):
    global point_1,point_2, drawing,flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            point_1 = (x,y)

        else:
            drawing = False
            flag = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            point_2 = (x,y)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",mouse_drawing)

# Features
sift = cv2.xfeatures2d.SIFT_create()

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_1 and point_2 and drawing:
        cv2.rectangle(frame,point_1,point_2,(0,255,0),1)

    #--------------------------------------Object detection / tracking------------------------------------------------------------
    if flag:
        reference_img = frame[point_1[1]:point_2[1],point_1[0]:point_2[0]]
        reference_gray_img = cv2.cvtColor(reference_img,cv2.COLOR_BGR2GRAY)

        kp_img, desc_img = sift.detectAndCompute(reference_gray_img,None)
        cv2.imshow("Reference image",reference_img)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flag = False
        enable = True

    if enable is True:
        kp_frame_gray, desc_frame_gray = sift.detectAndCompute(frame_gray, None)
        matches = flann.knnMatch(desc_img, desc_frame_gray, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        if len(good_points) > 10:   # statement if good_points > 10, object is detected

            # define query and train points
            query_pts = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_frame_gray[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

            # Find perspective matrix to define relationship between reference image and frame img
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matrix_inv = np.linalg.pinv(matrix)

            # Find object (destination points) on frame image according to perspective Matrix
            h, w, ch = reference_img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            # draw corner points
            dst_col = dst.ravel()
            dst_col = np.int32(dst_col)
            cv2.putText(frame,"1",(dst_col[0],dst_col[1]),cv2.FONT_ITALIC,1,(0,255,0),2)
            cv2.putText(frame, "2", (dst_col[2], dst_col[3]), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            cv2.putText(frame, "3", (dst_col[4], dst_col[5]), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            cv2.putText(frame, "4", (dst_col[6], dst_col[7]), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

            frame = cv2.polylines(frame, [np.int32(dst)], True, (0,255,0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()