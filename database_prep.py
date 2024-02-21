import sqlite3
import os
import numpy as np
import tensorflow as tf
from FRmodel import get_faceRecoModel
from config import database_path

def database():
    print("Put the image of the person in database folder")
    print("----------------------------------------------------------------------")
    name = input("Provide the person's name: ")

    image_path = os.path.join('./database/' + name + '.jpg')

    model = get_faceRecoModel()

    img = tf.image.decode_image(tf.io.read_file(image_path), dtype=tf.float32, expand_animations=False)
    img = tf.image.resize(img, (160,160))
    img = tf.expand_dims(img, axis=0)

    embeddings = model.predict(img)

    sum = tf.math.reduce_sum(embeddings)
    
    sum = float(np.array(sum, dtype=np.float32))
    
    return name.lower(), sum

name, sum_value = database()

conn = sqlite3.connect('face_database.db')
c = conn.cursor()

c.execute("""  CREATE TABLE IF NOT EXISTS face_data (
            name TEXT,
            encoding FLOAT
            ) """)

c.execute('INSERT INTO face_data (name, encoding) VALUES (?, ?)', (name, sum_value))

conn.commit()
conn.close()
    
print("Finished")