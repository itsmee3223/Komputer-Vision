import cv2
import os
import glob

# input positif
pathNegatif = './input/p/*.*'
# input negatif
path = './input/n/*.*'
# model training pertama
jam_classifier = cv2.CascadeClassifier('./classifier/cascade.xml');
# model training kedua
# jam_classifier = cv2.CascadeClassifier('./classifier (2)/cascade.xml');

# input positif
for filename in glob.glob(path):
    img = cv2.imread(filename)
    img = cv2.resize(img, (200, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    jam = jam_classifier.detectMultiScale(gray, 1.0485258, 6)
    if jam == ():
        print("Not found")
    for (x,y,w,h) in jam:
        cv2.rectangle(img, (x,y), (x+w,y+h), (127,0,255), 2)
        cv2.imshow('Detection', img)
        cv2.waitKey(0)
        basename = os.path.basename(filename)
        name = os.path.splitext(basename)[0]
        # silahkan uncomment line code 10 dan 28 serta comment line code 12 dan 30 untuk test model_1.
        #  lakukan hal sebaliknya untuk test model_2
        cv2.imwrite('./output/model_1/' + name + '_result.jpg', img)
        # cv2.imwrite('./output/model_2/' + name + '_result.jpg', img)

# input negatif
for filename in glob.glob(pathNegatif):
    img = cv2.imread(filename)
    img = cv2.resize(img, (200, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    jam = jam_classifier.detectMultiScale(gray, 1.0485258, 6)
    if jam == ():
        print("Not found")
    for (x,y,w,h) in jam:
        cv2.rectangle(img, (x,y), (x+w,y+h), (127,0,255), 2)
        cv2.imshow('Detection', img)
        cv2.waitKey(0)
        basename = os.path.basename(filename)
        name = os.path.splitext(basename)[0]
        # silahkan uncomment line code 10 dan 28 serta comment line code 12 dan 30 untuk test model_1.
        #  lakukan hal sebaliknya untuk test model_2
        cv2.imwrite('./output/model_1/' + name + '_result.jpg', img)
        # cv2.imwrite('./output/model_2/' + name + '_result.jpg', img)