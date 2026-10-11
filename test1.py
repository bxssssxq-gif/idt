import cv2
import numpy as np
import time  # เพิ่มการนำเข้า time สำหรับตั้งชื่อไฟล์

def empty(_): pass

cap = cv2.VideoCapture(0)

# หน้าต่างปรับสี
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 520, 240)
cv2.createTrackbar("H_min", "HSV", 40, 179, empty)
cv2.createTrackbar("H_max", "HSV", 80, 179, empty)
cv2.createTrackbar("S_min", "HSV", 50, 255, empty)
cv2.createTrackbar("S_max", "HSV", 255, 255, empty)
cv2.createTrackbar("V_min", "HSV", 50, 255, empty)
cv2.createTrackbar("V_max", "HSV", 255, 255, empty)

canvas = None
is_drawing = False
px, py = 0, 0 
drawing_color = (0, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hmin = cv2.getTrackbarPos("H_min", "HSV")
    hmax = cv2.getTrackbarPos("H_max", "HSV")
    smin = cv2.getTrackbarPos("S_min", "HSV")
    smax = cv2.getTrackbarPos("S_max", "HSV")
    vmin = cv2.getTrackbarPos("V_min", "HSV")
    vmax = cv2.getTrackbarPos("V_max", "HSV")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # ดูดสีจากวัตถุ
                obj_color = frame[cy, cx] 
                drawing_color = (int(obj_color[0]), int(obj_color[1]), int(obj_color[2]))

                cv2.circle(frame, (cx, cy), 10, drawing_color, -1)

                if is_drawing:
                    if px == 0 and py == 0: px, py = cx, cy
                    cv2.line(canvas, (px, py), (cx, cy), drawing_color, 10)
                    px, py = cx, cy
                    cv2.putText(frame, "STATUS: DRAWING", (10, 30), 1, 1.5, (0, 255, 0), 2)
                else:
                    px, py = 0, 0
                    cv2.putText(frame, "STATUS: PAUSED", (10, 30), 1, 1.5, (0, 0, 255), 2)

    combined = cv2.addWeighted(frame, 1, canvas, 0.8, 0)
    cv2.imshow("Mask", mask)
    cv2.imshow("Virtual Highlighter", combined)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("r"): is_drawing = True
    if k == ord("p"): is_drawing = False
    if k == ord("c"): canvas = np.zeros_like(frame)
    if k == ord("q"): break

cap.release()
cv2.destroyAllWindows()