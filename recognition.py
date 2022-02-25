# print version number; un comment when wanting to check
# print(cv2.__version__)
import cv2

cap = cv2.VideoCapture(0)



# we will now use cascade files that have pre-built objects,
# we can build our own however we will use ready samples for now
cascade_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_hand.xml')
#haarcascades/haarcascade_hand.xml
#haarcascades/haarcascade_eye.xml

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade_classifier.detectMultiScale(gray, 1.3, 5)


    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)

    # wait for 1 millisecond before going into the next frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()