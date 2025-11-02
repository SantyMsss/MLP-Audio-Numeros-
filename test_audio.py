import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import glob
import subprocess
import tempfile

def convert_to_wav(audio_path):
    """
    Convierte cualquier formato de audio a WAV temporal usando ffmpeg
    """
    file_ext = os.path.splitext(audio_path)[1].lower()
    
    # Si ya es WAV, retornar la ruta original
    if file_ext == '.wav':
        return audio_path, False
    
    # Convertir a WAV temporal usando ffmpeg
    try:
        print(f"🔄 Convirtiendo {file_ext} a WAV...")
        
        # Crear archivo temporal
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        # Buscar ffmpeg en ubicaciones comunes
        ffmpeg_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            "ffmpeg"  # Si está en PATH
        ]
        
        ffmpeg_cmd = None
        for path in ffmpeg_paths:
            if path == "ffmpeg" or os.path.exists(path):
                ffmpeg_cmd = path
                break
        
        if not ffmpeg_cmd:
            print("⚠️ FFmpeg no encontrado. Intentando usar librosa directamente...")
            return audio_path, False
        
        # Ejecutar ffmpeg
        cmd = [
            ffmpeg_cmd,
            '-i', audio_path,
            '-acodec', 'pcm_s16le',
            '-ar', '22050',
            '-ac', '1',
            '-y',
            temp_wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Conversión completada\n")
            return temp_wav_path, True
        else:
            print(f"⚠️ Error en ffmpeg, intentando con librosa...")
            return audio_path, False
            
    except Exception as e:
        print(f"⚠️ No se pudo convertir con ffmpeg: {e}")
        print("Intentando cargar directamente con librosa...")
        return audio_path, False

def extract_mfcc_features(y, sr, n_mfcc=13):
    """
    Extrae características MFCC del audio
    """
    # Extraer MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Calcular estadísticas (media por coeficiente)
    mfcc_features = np.mean(mfcc.T, axis=0)
    
    return mfcc_features

def predict_audio(model_path, audio_path):
    """
    Predice el número de un archivo de audio usando el modelo entrenado
    """
    print(f"\n{'='*60}")
    print(f"🎵 Analizando audio: {os.path.basename(audio_path)}")
    print(f"{'='*60}\n")
    
    temp_file_created = False
    temp_wav_path = None
    
    try:
        # Cargar modelo
        print("📦 Cargando modelo...")
        model = keras.models.load_model(model_path)
        print("✅ Modelo cargado exitosamente\n")
        
        # Convertir a WAV si es necesario
        wav_path, temp_file_created = convert_to_wav(audio_path)
        
        if wav_path is None:
            return None, 0
        
        temp_wav_path = wav_path if temp_file_created else None
        
        # Cargar y preprocesar audio
        print("🔊 Cargando archivo de audio...")
        y, sr = librosa.load(wav_path, mono=True, sr=None)
        print(f"✅ Audio cargado - Duración: {len(y)/sr:.2f}s, Sample Rate: {sr}Hz\n")
        
        # Extraer características MFCC
        print("🔍 Extrayendo características MFCC...")
        features = extract_mfcc_features(y, sr)
        print(f"✅ Características extraídas: {features.shape}\n")
        
        # Redimensionar para el modelo
        features = features.reshape(1, -1)
        
        # Predecir
        print("🤖 Realizando predicción...")
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        
        # Mostrar resultados
        print(f"\n{'='*60}")
        print(f"🎯 RESULTADO DE LA PREDICCIÓN")
        print(f"{'='*60}")
        print(f"🔢 Número predicho: {predicted_class}")
        print(f"📊 Confianza: {confidence:.2f}%")
        print(f"{'='*60}\n")
        
        # Mostrar todas las probabilidades
        print("📈 Probabilidades para cada clase:")
        for i, prob in enumerate(prediction[0]):
            bar = '█' * int(prob * 50)
            print(f"  {i}: {bar} {prob*100:5.2f}%")
        
        # Visualizar el espectrograma MFCC
        print("\n📊 Generando visualización...")
        mfcc_full = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        plt.figure(figsize=(14, 8))
        
        # Subplot 1: Forma de onda
        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
        plt.title(f'Forma de Onda - Archivo: {os.path.basename(audio_path)}', fontsize=14, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: MFCC
        plt.subplot(3, 1, 2)
        librosa.display.specshow(mfcc_full, x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Coeficientes MFCC', fontsize=14, fontweight='bold')
        plt.ylabel('Coeficiente MFCC')
        
        # Subplot 3: Espectrograma
        plt.subplot(3, 1, 3)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='hz', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Espectrograma', fontsize=14, fontweight='bold')
        plt.ylabel('Frecuencia (Hz)')
        
        plt.suptitle(f'Predicción: {predicted_class} | Confianza: {confidence:.2f}%', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error procesando audio: {e}")
        import traceback
        traceback.print_exc()
        return None, 0
    
    finally:
        # Limpiar archivo temporal si se creó
        if temp_file_created and temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
                print("🗑️ Archivo temporal eliminado")
            except:
                pass

def main():
    # Rutas
    MODEL_PATH = r"C:\Users\USER\Desktop\ING SISTEMAS\7\IA\MLP\mlp_digit_classifier.h5"
    PRUEBAS_PATH = r"C:\Users\USER\Desktop\ING SISTEMAS\7\IA\MLP\pruebas"
    
    # Verificar que existe el modelo
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo en: {MODEL_PATH}")
        print("   Por favor, entrena el modelo primero ejecutando app.py")
        return
    
    # Verificar que existe la carpeta de pruebas
    if not os.path.exists(PRUEBAS_PATH):
        print(f"❌ No se encontró la carpeta: {PRUEBAS_PATH}")
        print("   Por favor, crea la carpeta 'pruebas' y agrega tus archivos de audio")
        return
    
    # Buscar archivos de audio en la carpeta pruebas
    audio_extensions = ['*.m4a', '*.wav', '*.mp3', '*.flac', '*.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(PRUEBAS_PATH, ext)))
    
    if not audio_files:
        print(f"❌ No se encontraron archivos de audio en: {PRUEBAS_PATH}")
        print(f"   Formatos soportados: {', '.join(audio_extensions)}")
        return
    
    print(f"\n🎵 Se encontraron {len(audio_files)} archivo(s) de audio:\n")
    for i, file in enumerate(audio_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # Probar con cada archivo
    print(f"\n{'='*60}")
    print("Iniciando pruebas...")
    print(f"{'='*60}")
    
    for audio_file in audio_files:
        predicted, confidence = predict_audio(MODEL_PATH, audio_file)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
