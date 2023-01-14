import cv2 as cv

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    jam = jam_cascade.detectMultiScale(frame_gray, 1.0315258, 6)
    for (x, y, w, h) in jam:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv.imshow("Caputre - jam detection", frame)

# model training pertama
jam_cascade_name = "./classifier/cascade.xml"
# model training kedua
jam_cascade_name_2 = "./classifier (2)/cascade.xml"

jam_cascade = cv.CascadeClassifier()
jam_cascade.load(cv.samples.findFile(jam_cascade_name_2))

cap = cv.VideoCapture(0)
if not cap.isOpened:
    print("--(!)Error opening video capture")
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print("--(!) No captured frame -- Break!")
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break