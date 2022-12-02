import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)

data_image = fr.load_image_file("ogo.jpg")
data_face_encoding = fr.face_encodings(data_image)[0]

known_face_encondings = [data_face_encoding]
known_face_names = ["OGO"]

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        esleme = fr.compare_faces(known_face_encondings, face_encoding)

        name = "TANIMSIZ"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        eslemeindex = np.argmin(face_distances)
        if esleme[eslemeindex]:
            name = known_face_names[eslemeindex]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 12, bottom - 12), font, 1.1, (255, 255, 255), 1)

    cv2.imshow('Yuz Tanıma', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()