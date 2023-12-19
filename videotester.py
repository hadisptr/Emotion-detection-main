#  Import library yang diperlukan
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

# Inisialisasi detektor wajah menggunakan Haar Cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Inisialisasi jumlah emosi yang terdeteksi
emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

# Daftar emosi
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Menunggu pengguna menekan tombol Enter
# input("Tekan Enter untuk memulai deteksi emosi...")

# Mengambil 10 ekspresi wajah
expressions_captured = 0
while expressions_captured < 20:
    # Membaca frame dari kamera
    ret, test_img = cap.read()
    if not ret:
        continue

    # Mengonversi frame ke skala abu-abu
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Mendeteksi wajah menggunakan Haar Cascade
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        # Mendapatkan area wajah
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Memprediksi emosi menggunakan model
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # Memperbarui jumlah emosi yang terdeteksi
        emotion_counts[predicted_emotion] += 1

        # Menampilkan hasil prediksi pada frame
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Meresize frame dan menampilkannya
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Analisis emosi wajah', resized_img)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(10) == ord('q'):
        break

    # Cek apakah sudah terdeteksi 10 ekspresi wajah
    if sum(emotion_counts.values()) == 20:
        break

# Menutup kamera video dan jendela tampilan
cap.release()
cv2.destroyAllWindows()

# Mencari emosi dengan jumlah terbanyak
highest_emotion = max(emotion_counts, key=emotion_counts.get)

# Memutar musik berdasarkan emosi dengan jumlah terbanyak
music_folder = 'music'
music_file = os.path.join(music_folder, highest_emotion + '.mp3')

pygame.mixer.init()
pygame.mixer.music.load(music_file)
pygame.mixer.music.play()

# Membuat diagram batang untuk distribusi emosi
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

# Menunggu musik selesai diputar
pygame.mixer.music.stop()
pygame.mixer.quit()
