import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from moviepy.editor import VideoFileClip

# Fonction pour extraire les images de la vidéo
def extract_frames(video_path, num_frames):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frames = []
    for t in np.linspace(0, duration, num_frames, endpoint=False):
        frame = clip.get_frame(t)
        frame = (frame * 255).astype(np.uint8)  # Convertir les pixels en valeurs d'entiers
        frame = frame[:, :, :3]  # Supprimer le canal alpha s'il existe
        frames.append(frame)
    clip.close()
    return frames

# Chargement des données
def load_data(video_dir):
    X = []
    y = []
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            age = int(video_file.split('.')[0])  # L'âge est extrait du nom de la vidéo
            video_path = os.path.join(video_dir, video_file)
            frames = extract_frames(video_path, num_frames=10)
            if len(frames) == 10:
                X.extend(frames)
                y.extend([age] * 10)  # Répéter l'âge pour chaque frame de la vidéo
    X = np.array(X) / 255.0  # Normaliser les pixels
    y = np.array(y)
    return X, y

# Construction du modèle
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    return model

# Entraînement du modèle
def train_model(model, X_train, y_train, epochs=10):
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)

# Chargement des données d'entraînement
video_dir = 'video'  # Répertoire contenant les vidéos
X_train, y_train = load_data(video_dir)
if len(X_train) == 0:
    print("Aucune vidéo valide trouvée dans le répertoire.")
    exit()

# Création et entraînement du modèle
input_shape = X_train[0].shape
model = build_model(input_shape)
train_model(model, X_train, y_train)

# Évaluation du modèle (optionnel)
# Si vous avez des données de validation ou de test, vous pouvez les charger et évaluer le modèle ici.

# Sauvegarde du modèle (optionnel)
model.save("age_prediction_model.h5")
print("Modèle sauvegardé avec succès.")

