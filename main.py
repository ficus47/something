from tensorflow.keras import layers
import tensorflow as tf

import cv2
import os

def preprocess_video(video_path, output_folder, frame_width, frame_height):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo.")
        return

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    # Lire chaque frame de la vidéo
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionner le frame
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Sauvegarder le frame prétraité
        output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # Fermer la vidéo
    cap.release()
    cv2.destroyAllWindows()


model = tf.keras.models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

