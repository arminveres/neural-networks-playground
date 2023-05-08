import cv2
webcam = cv2.VideoCapture(0) # try different number if not working

while True:
    check, frame = webcam.read()
    cv2.imshow("frame", frame)
    if not check:
        break
    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
