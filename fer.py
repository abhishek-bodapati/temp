import cv2
import time
import model

cv2.namedWindow("preview")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly if not cap. isOpened():
if not cap.isOpened():
    cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read() ## read one image from a video
    result = model.analyze(frame, actions = ['emotion'], enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the faces for (x, y, w, h) in faces:
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for 
    # inserting text on video

    cv2.putText(
        frame,
        result['dominant_emotion'],
        (50, 50),
        font,
        3,
        (0, 255, 255),
        2,
        cv2.LINE_4
    )

    cv2.imshow ('Original video', frame)
    if cv2.waitKey (2) & 0xFF == ord('q'):
        break
    time.sleep(2)

cap.release()
cv2.destroyAllWindows()
