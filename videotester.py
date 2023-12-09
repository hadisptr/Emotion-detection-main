import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
from keras.models import load_model
import matplotlib.pyplot as plt
import time
import pygame

warnings.filterwarnings("ignore")

# Load the model
model = load_model("best_model.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Initialize emotion counts
emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Capture 10 facial expressions
expressions_captured = 0
while expressions_captured < 10:
    ret, test_img = cap.read()
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # Update emotion counts
        emotion_counts[predicted_emotion] += 1

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Analisis emosi wajah', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

    # Check if 10 expressions have been captured
    if sum(emotion_counts.values()) == 10:
        break

cap.release()
cv2.destroyAllWindows()

# Find the emotion with the highest count
highest_emotion = max(emotion_counts, key=emotion_counts.get)

# Play music based on the highest emotion
music_folder = 'music'
music_file = os.path.join(music_folder, highest_emotion + '.mp3')

pygame.mixer.init()
pygame.mixer.music.load(music_file)
pygame.mixer.music.play()

# Create a bar chart for emotion distribution
emotions = list(emotion_counts.keys())
counts = list(emotion_counts.values())

plt.bar(emotions, counts)
plt.title('Distribusi Emosi')
plt.xlabel('Emosi')
plt.ylabel('Jumlah')

# Simpan grafik sebagai gambar
plt.savefig('distribusi_emosi.png')

# Tampilkan grafik
plt.show()

# Wait for music to finish playing
pygame.mixer.music.stop()
pygame.mixer.quit()
