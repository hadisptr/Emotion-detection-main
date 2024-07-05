import cv2

# Inisialisasi detektor wajah menggunakan Haar Cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        continue

    # Mengonversi frame ke skala abu-abu
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah menggunakan Haar Cascade
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        # Menggambar kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Menampilkan frame dengan wajah yang terdeteksi
    cv2.imshow('Deteksi Wajah', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera video dan jendela tampilan
cap.release()
cv2.destroyAllWindows()
