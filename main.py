import cv2
import mediapipe as mp
import face_recognition as FR

imgDir = './44_barack_obama_family.jpeg'

img = cv2.imread(imgDir)

img2 = cv2.imread(r'./Barack.jpeg')

H,W, _ = img.shape

mpFaceDetection = mp.solutions.face_detection

barackFace=FR.load_image_file(r'C:\Users\pande\Documents\Python\demolmages\known\President_Barack_Obama.jpg')
faceLoc=FR.face_locations(barackFace)[0]
barackFaceEncode=FR.face_encodings(barackFace)[0]

knownEncodings=[barackFaceEncode]
names=['President Obama']
font=cv2.FONT_HERSHEY_SIMPLEX
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
while True:
# with mpFaceDetection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as FaceDetection:
#     imgColor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     out = FaceDetection.process(imgColor)

    ignore, unknownFace = cam.read()
    unknownFace = FR.load_image_file(r'./Barack.jpeg')
    unknownFaceRGB = cv2.cvtColor(unknownFace, cv2.COLOR_BGR2RGB)
    faceLocations = FR.face_locations(unknownFaceRGB)
    unknownEncodings = FR.face_encodings(unknownFaceRGB, faceLocations)

    # if out.detections is not None:
        # for detection in out.detections:
    for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
        # location_data = unknownEncoding.location_data
        # bbox = location_data.relative_bounding_box

        matches = FR.compare_faces(knownEncodings, unknownEncoding)
        top, right, bottom, left = faceLocation
        print(faceLocation)
        cv2.rectangle(unknownFace, (left, top), (right, bottom), (255, 0, 0), 3)
        name = 'irrelevant'
        if True in matches:
            # x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            #
            # x1 = int(x1 * W)
            # y1 = int(y1 * H)
            # w = int(w * W)
            # h = int(h * H)
            #
            # img[y1:y1+h,x1:x1+w,:] = cv2.blur(img[y1:y1+h,x1:x1+w,:], (30,30))
            matchIndex = matches.index(True)
            print(matchIndex)
            print(names[matchIndex])
            name = names[matchIndex]
        else:
            img2[top:bottom, left:right] = cv2.blur(img2[top:bottom, left:right], (50, 50))
        # cv2.putText(unknownFace, name, (left, top), font, .75, (0, 0, 255), 2)
    resized_img = cv2.resize(img2, (640, 480))
    cv2.imshow('resized_img', resized_img)
    if cv2.waitKey(0) & 0xff == ord('q'):
        break
    width = 1920
    height = 1080
    dsize = (width, img.shape[0])


    # cv2.imshow('resized_img', resized_img)
    # cv2.waitKey(0)

cv2.imwrite('./BarackBlur.jpg', resized_img)

cam.release()
cv2.destroyAllWindows()