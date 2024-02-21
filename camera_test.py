import cv2

cap = cv2.VideoCapture('./test_video.mp4')

while cap.isOpened():
    success, frame = cap.read()
    print(frame.shape)

    cv2.imshow("frame", frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()