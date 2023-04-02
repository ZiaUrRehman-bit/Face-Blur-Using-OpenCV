import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fd

cTime = 0
pTime = 0

cam = cv2.VideoCapture(0)
detection = fd.faceDetect()

while True:
    Success, frame = cam.read()
    frame, bboxs = detection.findFace(frame)
    # print(bboxs)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    if bboxs:
       
        x, y, w, h = bboxs
        imgCrop = frame[y:y+h, x:x+w]
        imgBlur = cv2.blur(imgCrop, (35, 35))
        frame[y:y+h, x:x+w] = imgBlur
        # cv2.imshow(f"image croped", imgCrop)

    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
