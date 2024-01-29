import tensorflow as tf
import cv2
import os
import time
import sqlite3
from Face_detector_cascade import faceDetector
from config import image_shape, THRESHOLD
from fr_utils import get_faceRecoModel

conn = sqlite3.connect('face_database.db')
c = conn.cursor()

def verify(roi, model, identity):
    emb = model.predict(roi)
    emb = tf.math.reduce_sum(emb)
    
    p = c.execute(f"SELECT * FROM face_data WHERE name=?", (identity))
    person = p.fetchone()
    
    if person is not None:
        stored_emb = tf.constant(person[1])
        distance = tf.math.reduce_sum(stored_emb - emb)
        if distance < THRESHOLD:
            return True
        else:
            return False
    else:
        return False

def live_face_detection():
    detector = faceDetector("haarcascade_frontalface_default.xml")
    
    cap = cv2.VideoCapture(0)
    
    identity = input("Name of the person: ")
    print("Verifying...")
    identity = identity.lower()
    
    model = get_faceRecoModel()    

    time.sleep(3)
    
    while cap.isOpened():
        succes, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect(image=gray)
        
        for (x,y,w,h) in faces:
            roi = frame[y:y+h,x:x+w]
            roi = tf.image.resize(roi, (160,160))

            res = verify(roi, model, identity)
            if res:
                return res, identity
        
        cv2.imshow("video", frame)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

def face_verification():
    res = False
    res, identity = live_face_detection()
    door_open = False
    
    if res:
        print(f"Welcome {identity}")
        door_open = True
    else:
        print("Wrong person")
        
if __name__ == "__main__":
    face_verification()