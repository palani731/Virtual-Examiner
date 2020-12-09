import cv2

videocaptureobj = cv2.VideoCapture(0)
result = True
while(result):
    clar,frame = videocaptureobj.read()
    cv2.imwrite("image.jpg",frame)
    result = False
videocaptureobj.release()
cv2.destroyAllWindows()