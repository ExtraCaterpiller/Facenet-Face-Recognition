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


def distance(live_emb, stored_emb):
    dis = tf.math.reduce_sum(live_emb, stored_emb)
    return dis

def recognizer(roi, model):
    emb = model.predict(roi)
    emb = tf.math.reduce_sum(emb)
    
    p = c.execute("SELECT * FROM face_data")
    persons = p.fetchall()
    
    minimum_dis = float('inf')
    identity = ''
    if persons is not None:
        for person in persons:
            stored_emb = tf.constant(person[1])
            dis = distance(emb, stored_emb)
            
            if dis < minimum_dis:
                minimum_dis = dis
                identity = person[0]
        return minimum_dis, identity
    else:
        print("Database empty")
        return float("inf"), ''

def live_face_detection():
    detector = faceDetector("haarcascade_frontalface_default.xml")
    
    cap = cv2.VideoCapture(0)

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('output.mp4', codec, 20.0, (1280, 720))
    
    model = get_faceRecoModel()

    time.sleep(3)
    
    while cap.isOpened():
        succes, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect(image=gray)
        
        face_reconized = False
        
        for (x,y,w,h) in faces:
            roi = frame[y:y+h,x:x+w]
            roi = tf.image.resize(roi, (image_shape[0], image_shape[1]))
            dis, identity = recognizer(roi, model)
            if dis < THRESHOLD:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, text=identity, org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,255), thickness=1.5)
                face_reconized = True
                
        if face_reconized:
            frame = cv2.resize(frame, (1280,720))
            output.write(frame)
        
        cv2.imshow("video", frame)
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    output.release()
    cap.release()
    cv2.destroyAllWindows()      

if __name__ == "__main__":
    live_face_detection()