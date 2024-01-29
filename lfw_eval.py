import sqlite3
from config import image_shape, THRESHOLD
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score
from fr_utils import get_faceRecoModel

model = get_faceRecoModel()

def preprocess(path):
    img = tf.io.decode_image(tf.io.read_file(path), dtype=tf.float32, expand_animations=False)
    img = tf.image.resize(img, (image_shape[0], image_shape[1]))
    return img

def prep_eval_db(names):
    conn = sqlite3.connect("eval_embedding_data.db")
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding FLOAT,
            name TEXT
        )
    ''')

    for name in names:
        d = os.path.join("./lfw/" + name)
        print(len(os.listdir(d)))
        for i in os.listdir(d):
            print(i)
            print(name)
            image = preprocess(d+'/'+i)
            image = tf.expand_dims(image, axis=0)
            embedding = model.predict(image)
            emb = float(tf.math.reduce_sum(embedding))
            print(emb)
            c.execute('INSERT INTO embeddings (name, embedding) VALUES (?, ?)', (name, emb))

    conn.commit()
    conn.close()
    
def lfw_eval():
    conn = sqlite3.connect("eval_embedding_data.db")
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM embeddings')
    total_entries = c.fetchone()[0]
    
    print(f'Total entries: {total_entries}')
    
    result = []
    
    c.execute('SELECT * FROM embeddings')
    all_entries = c.fetchall()

    for i in range(total_entries):
        anchor_entry = all_entries[i]
        
        for j in range(total_entries-i-1):
            comparison_entry = all_entries[i + j + 1]
            
            distance = abs(anchor_entry[1]-comparison_entry[1])
            if distance < THRESHOLD and anchor_entry[2] == comparison_entry[2]:
                result.append(1)
            elif distance > THRESHOLD and anchor_entry[2] != comparison_entry[2]:
                result.append(1)
            else:
                result.append(0)
    
    conn.commit()
    conn.close()
                
    y_true = np.ones(len(result), dtype=np.uint8)
    
    accuracy = accuracy_score(y_true, result)
    print(f"Accuracy: {accuracy} for threshold: {THRESHOLD}")
    
 
names = os.listdir('./lfw') 
        
if __name__ == "__main__":
    #prep_eval_db(names)
    lfw_eval()