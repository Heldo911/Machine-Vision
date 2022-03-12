import  cv2
import  numpy as np

point_1 = ()
point_2 = ()
window_point_1 = ()
window_point_2 = ()
flag = False
parameter = False
drawing = False
green_img = np.zeros((480, 640, 3), np.uint8)
green_img[0:480, 0:640] = (0, 255, 0)
kernel = np.ones((3,3),np.uint8)

cap = cv2.VideoCapture(1)

def mouse_drawing(event,x,y,flags,params):
    global point_1,point_2, window_point_1, window_point_2, drawing,flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            point_1 = (x,y)

        else:
            drawing = False
            window_point_1 = point_1
            window_point_2 = point_2
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
    if flag is True:
        reference_img = frame[point_1[1]:point_2[1],point_1[0]:point_2[0]]
        reference_gray_img = cv2.cvtColor(reference_img,cv2.COLOR_BGR2GRAY)
        reference_canny_img = cv2.Canny(reference_img, 50, 255)
        kp_img, desc_img = sift.detectAndCompute(reference_gray_img,None)
        cv2.imshow("Reference image",reference_img)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flag = False
        parameter = True

    if parameter is True:

        kp_frame_gray, desc_frame_gray = sift.detectAndCompute(frame_gray, None)

        if len(kp_frame_gray) > 1:
            matches = flann.knnMatch(desc_img, desc_frame_gray, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            if len(good_points) > 10:   # statement if good_points > 10, object is detected

                # Homography / Perspective transform
                query_pts = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32([kp_frame_gray[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

                # Find perspective matrix to define relationship between reference image and frame img
                matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                matrix_inv = np.linalg.pinv(matrix)

                # Find object (destination points) on frame image according to perspective Matrix
                h, w, ch = reference_img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)

                #--------------------------------Comparison detected object vs reference object-----------------------------
                #Tranfrorm detected object from frame perspective to reference perspective
                warp_frame_img = cv2.warpPerspective(frame,matrix_inv,(w,h))
                warp_frame_canny_img = cv2.Canny(warp_frame_img,50,255)
                warp_frame_dilate_img = cv2.dilate(warp_frame_canny_img,kernel,iterations=1)

                #match between reference object and detected object
                bit_and_warp_reference = cv2.bitwise_and(reference_canny_img,warp_frame_dilate_img)

                #transform match from reference perspective to frame perspective
                mask = cv2.warpPerspective(bit_and_warp_reference, matrix, (640,480))
                mask_inv = cv2.bitwise_not(mask)

                #---------------------------------Display of results--------------------------------------------------------
                frame_1 = cv2.bitwise_and(frame,frame,mask=mask)
                frame_1_inv = cv2.bitwise_and(frame, frame, mask=mask_inv)
                difference = cv2.bitwise_and(green_img,frame_1)
                result = cv2.add(frame_1_inv, difference)

                frame = cv2.polylines(result, [np.int32(dst)], True, (255,0,0), 2)
                cv2.imshow("warp_frame_img", warp_frame_img)
                cv2.imshow("warp_frame_img_canny",warp_frame_canny_img)
                cv2.imshow("reference_img_canny", reference_canny_img)
                cv2.imshow("bit_and_warp_reference", bit_and_warp_reference)
            #   cv2.imshow("warp_frame_dilate_img", warp_frame_dilate_img)

                font = cv2.FONT_ITALIC
                match = np.sum(bit_and_warp_reference==255)/np.sum(reference_canny_img==255)*100
                cv2.putText(frame,"Match = " + str(int(match)) + "%",(400, 30), font, 1, (0, 255, 0),2)

            else:
                cv2.putText(frame, "Match = 0" + "%", (400, 30), font, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()