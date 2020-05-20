import cv2 as cv


def face_detect(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("imgs/haarcascades/haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.02, 1)
    for x, y, w, h in faces:
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imshow("Face detect image", src)


src = cv.imread("imgs/science.jpeg")
cv.namedWindow("Face detect image", cv.WINDOW_AUTOSIZE)
face_detect(src)
cv.waitKey(0)
cv.destroyAllWindows()
