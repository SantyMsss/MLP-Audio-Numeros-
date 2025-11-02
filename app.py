import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 1. Cargar y preprocesar el dataset
def load_dataset(dataset_path):
    """
    Carga el dataset de audios y extrae características MFCC
    """
    features = []
    labels = []
    
    # Mapeo de números en español
    number_map = {
        'zero': 0, 'cero': 0,
        'one': 1, 'uno': 1,
        'two': 2, 'dos': 2,
        'three': 3, 'tres': 3,
        'four': 4, 'cuatro': 4,
        'five': 5, 'cinco': 5,
        'six': 6, 'seis': 6,
        'seven': 7, 'siete': 7,
        'eight': 8, 'ocho': 8,
        'nine': 9, 'nueve': 9,
        'ten': 10, 'diez': 10
    }
    
    # Buscar archivos de audio
    audio_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
    
    print(f"Encontrados {len(audio_files)} archivos de audio")
    
    for audio_file in audio_files:
        try:
            # Extraer etiqueta del nombre del archivo
            filename = os.path.basename(audio_file).lower()
            
            # Buscar el número en el nombre del archivo
            label = None
            for num_name, num_value in number_map.items():
                if num_name in filename:
                    label = num_value
                    break
            
            if label is None:
                continue
            
            # Cargar audio
            y, sr = librosa.load(audio_file, mono=True, sr=None)
            
            # Extraer características MFCC
            mfcc_features = extract_mfcc_features(y, sr)
            
            features.append(mfcc_features)
            labels.append(label)
            
        except Exception as e:
            print(f"Error procesando {audio_file}: {e}")
            continue
    
    return np.array(features), np.array(labels)

def extract_mfcc_features(y, sr, n_mfcc=13):
    """
    Extrae características MFCC del audio
    """
    # Extraer MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Calcular estadísticas (media por coeficiente)
    mfcc_features = np.mean(mfcc.T, axis=0)
    
    return mfcc_features

# 2. Crear el modelo MLP
def create_mlp_model(input_shape, num_classes):
    """
    Crea un modelo MLP para clasificación de audio
    """
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 3. Función para predecir un nuevo audio
def predict_audio(model, audio_path, label_encoder):
    """
    Predice el número de un archivo de audio
    """
    try:
        # Cargar y preprocesar audio
        y, sr = librosa.load(audio_path, mono=True, sr=None)
        features = extract_mfcc_features(y, sr)
        
        # Redimensionar para el modelo
        features = features.reshape(1, -1)
        
        # Predecir
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None, 0

# 4. Entrenamiento y evaluación
def main():
    # Configuración
    DATASET_PATH = r"C:\Users\USER\Desktop\ING SISTEMAS\7\IA\MLP\content\digit_dataset"  # Ajusta esta ruta
    MODEL_SAVE_PATH = "mlp_digit_classifier.h5"
    
    # Verificar si existe el directorio
    if not os.path.exists(DATASET_PATH):
        print(f"❌ No se encontró el directorio: {DATASET_PATH}")
        print("\n📁 Por favor, crea una carpeta 'digit_dataset' y agrega archivos de audio (.wav)")
        print("   Los archivos deben tener el nombre del número en su nombre, por ejemplo:")
        print("   - cero_01.wav, cero_02.wav")
        print("   - uno_01.wav, uno_02.wav")
        print("   - dos_01.wav, etc.")
        print("\n💡 O puedes cambiar DATASET_PATH a la ubicación de tus archivos de audio.")
        return
    
    # Cargar dataset
    print("Cargando dataset...")
    X, y = load_dataset(DATASET_PATH)
    
    if len(X) == 0:
        print("No se encontraron archivos de audio. Verifica la ruta del dataset.")
        return
    
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Distribución de clases: {np.unique(y, return_counts=True)}")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Crear modelo
    input_shape = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"Input shape: {input_shape}, Número de clases: {num_classes}")
    
    model = create_mlp_model(input_shape, num_classes)
    
    # Resumen del modelo
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Entrenar modelo
    print("Entrenando modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar modelo
    print("Evaluando modelo...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión en prueba: {test_accuracy:.4f}")
    
    # Guardar modelo
    model.save(MODEL_SAVE_PATH)
    print(f"Modelo guardado como {MODEL_SAVE_PATH}")
    
    # Graficar resultados
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión validación')
    plt.title('Precisión del modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Probar con un audio específico
    print("\n--- Probando clasificación ---")
    
    # Buscar un archivo de prueba
    test_files = glob.glob(os.path.join(DATASET_PATH, "**/*.wav"), recursive=True)
    if test_files:
        test_file = test_files[0]
        print(f"Probando con: {test_file}")
        
        predicted, confidence = predict_audio(model, test_file, None)
        
        if predicted is not None:
            print(f"Predicción: {predicted} (confianza: {confidence:.4f})")
            
            # Mostrar espectrograma
            y, sr = librosa.load(test_file, mono=True, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar()
            plt.title(f'MFCC - Predicción: {predicted} (Confianza: {confidence:.2f})')
            plt.show()

# 5. Función para usar el modelo entrenado
def load_and_predict(model_path, audio_path):
    """
    Carga un modelo entrenado y predice un audio
    """
    model = keras.models.load_model(model_path)
    
    predicted, confidence = predict_audio(model, audio_path, None)
    
    if predicted is not None:
        print(f"🔊 Audio: {audio_path}")
        print(f"🔢 Número predicho: {predicted}")
        print(f"📊 Confianza: {confidence:.4f}")
        return predicted
    else:
        print("Error en la predicción")
        return None

# Ejecutar entrenamiento
if __name__ == "__main__":
    main()

# Para usar el modelo entrenado posteriormente:
# load_and_predict("mlp_digit_classifier.h5", "ruta/a/tu/audio.wav")