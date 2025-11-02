# 🎙️ Clasificador de Dígitos por Audio con MLP

Proyecto de Inteligencia Artificial que implementa un clasificador de números hablados (0-9) utilizando una Red Neuronal Multicapa (MLP) con TensorFlow/Keras.

## 📋 Descripción

Este proyecto utiliza técnicas de procesamiento de señales de audio y aprendizaje profundo para reconocer dígitos hablados en archivos de audio. El sistema extrae características MFCC (Mel-Frequency Cepstral Coefficients) de los audios y las utiliza para entrenar un modelo de red neuronal que puede clasificar números del 0 al 9.

## 🚀 Características Principales

- ✅ **Clasificación de dígitos del 0 al 9** en español e inglés
- ✅ **Extracción de características MFCC** para representación del audio
- ✅ **Red Neuronal Multicapa (MLP)** con 5 capas y dropout
- ✅ **Visualización de resultados** con gráficos de precisión y pérdida
- ✅ **Análisis espectral** con espectrogramas MFCC
- ✅ **Predicción en tiempo real** con nuevos archivos de audio
- ✅ **Soporte para múltiples formatos** de audio (WAV, M4A, MP3, etc.)

## 📁 Estructura del Proyecto

```
MLP/
├── app.py                      # Script principal de entrenamiento
├── test_audio.py              # Script para probar el modelo con nuevos audios
├── create_test_audio.py       # Genera audio sintético para pruebas
├── convert_m4a_to_wav.py      # Instrucciones para convertir formatos
├── mlp_digit_classifier.h5    # Modelo entrenado guardado
├── content/
│   └── digit_dataset/         # Dataset de entrenamiento (5000 archivos)
├── pruebas/                   # Carpeta para archivos de audio de prueba
└── README.md                  # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **Python 3.11**
- **TensorFlow 2.20.0** - Framework de deep learning
- **Keras 3.12.0** - API de alto nivel para redes neuronales
- **Librosa 0.11.0** - Procesamiento y análisis de audio
- **NumPy 2.3.4** - Computación numérica
- **Matplotlib 3.10.7** - Visualización de datos
- **Scikit-learn 1.7.2** - Preprocesamiento y división de datos

## 📦 Instalación

1. **Clonar o descargar el proyecto**

2. **Crear entorno virtual**
```powershell
python -m venv .venv
```

3. **Activar el entorno virtual**
```powershell
.\.venv\Scripts\Activate.ps1
```

4. **Instalar dependencias**
```powershell
pip install tensorflow librosa matplotlib scikit-learn pydub scipy
```

## 📊 Dataset

El proyecto utiliza un dataset de 5000 archivos de audio de dígitos hablados:
- **Formato**: WAV
- **Contenido**: Números del 0 al 9
- **Idiomas**: Español e inglés
- **Distribución**: ~500 muestras por clase (balanceado)

### Estructura del Dataset:
```
content/digit_dataset/
├── zero_en_M_1.wav
├── one_es_F_1.wav
├── two_en_M_2.wav
├── ...
```

Los archivos deben contener el nombre del número en su nombre de archivo (ej: "cero", "uno", "zero", "one", etc.)

## 🎯 app.py - Script Principal de Entrenamiento

### Funcionalidad

Este script implementa todo el pipeline de entrenamiento del modelo:

#### 1. **Carga y Preprocesamiento de Datos**
```python
load_dataset(dataset_path)
```
- Busca recursivamente archivos `.wav` en el dataset
- Identifica el número hablado desde el nombre del archivo
- Mapea nombres en español e inglés a valores numéricos (0-9)
- Procesa 4999 de 5000 archivos exitosamente

#### 2. **Extracción de Características MFCC**
```python
extract_mfcc_features(y, sr, n_mfcc=13)
```
- Extrae 13 coeficientes MFCC de cada audio
- Calcula la media de cada coeficiente a lo largo del tiempo
- Genera un vector de características de 13 dimensiones por audio

#### 3. **Arquitectura del Modelo MLP**
```python
create_mlp_model(input_shape=13, num_classes=10)
```

**Capas de la red:**
- **Capa 1**: Dense(256) + ReLU + Dropout(0.3)
- **Capa 2**: Dense(128) + ReLU + Dropout(0.3)
- **Capa 3**: Dense(64) + ReLU + Dropout(0.2)
- **Capa 4**: Dense(32) + ReLU
- **Capa 5**: Dense(10) + Softmax (salida)

**Total de parámetros**: 47,146 (184.16 KB)

#### 4. **Configuración de Entrenamiento**
- **Optimizador**: Adam
- **Función de pérdida**: Sparse Categorical Crossentropy
- **Métricas**: Accuracy
- **Callbacks**: 
  - EarlyStopping (paciencia: 10 épocas)
  - ReduceLROnPlateau (reduce learning rate en mesetas)

#### 5. **División de Datos**
- **Entrenamiento**: 60% (3,199 muestras)
- **Validación**: 20% (800 muestras)
- **Prueba**: 20% (1,000 muestras)
- **Estratificación**: Sí (mantiene proporción de clases)

#### 6. **Resultados del Entrenamiento**
- ✅ **Precisión en validación**: 100% (desde época 48)
- ✅ **Precisión en prueba**: 100%
- ✅ **Épocas totales**: 100
- ✅ **Learning rate final**: 0.00025

#### 7. **Visualizaciones Generadas**
1. Gráfico de precisión (entrenamiento vs validación)
2. Gráfico de pérdida (entrenamiento vs validación)
3. Espectrograma MFCC de un audio de prueba
4. Predicción de ejemplo con nivel de confianza

#### 8. **Modelo Guardado**
```
mlp_digit_classifier.h5
```
Formato HDF5 compatible con TensorFlow/Keras

### Ejecución

```powershell
python app.py
```

**Salida esperada:**
```
Cargando dataset...
Encontrados 5000 archivos de audio
Dataset cargado: 4999 muestras, 13 características
Entrenando modelo...
Epoch 100/100
Precisión en prueba: 1.0000
Modelo guardado como mlp_digit_classifier.h5
```

---

## 🧪 test_audio.py - Script de Predicción

### Funcionalidad

Este script permite probar el modelo entrenado con nuevos archivos de audio:

#### 1. **Carga del Modelo Entrenado**
```python
model = keras.models.load_model(model_path)
```

#### 2. **Conversión de Formatos**
```python
convert_to_wav(audio_path)
```
- Detecta archivos que no son WAV
- Intenta convertir usando FFmpeg
- Soporta formatos: M4A, MP3, FLAC, OGG

#### 3. **Procesamiento de Audio**
- Carga el archivo con Librosa
- Convierte a mono si es estéreo
- Extrae características MFCC (13 coeficientes)
- Normaliza las características

#### 4. **Predicción**
```python
prediction = model.predict(features)
```
- Genera probabilidades para cada clase (0-9)
- Identifica la clase con mayor probabilidad
- Calcula el nivel de confianza

#### 5. **Visualización Detallada**

**Gráfico 1: Forma de Onda**
- Muestra la amplitud del audio en el tiempo
- Duración total del audio

**Gráfico 2: Coeficientes MFCC**
- Visualización del espectrograma MFCC
- 13 coeficientes a lo largo del tiempo

**Gráfico 3: Espectrograma de Frecuencias**
- Análisis espectral completo
- Frecuencias vs tiempo

#### 6. **Reporte de Resultados**
```
🎯 RESULTADO DE LA PREDICCIÓN
🔢 Número predicho: 8
📊 Confianza: 100.00%

📈 Probabilidades para cada clase:
  0:   0.00%
  1:   0.00%
  ...
  8: ██████████████████████████████ 100.00%
  9:   0.00%
```

### Ejecución

```powershell
python test_audio.py
```

**Proceso:**
1. Busca archivos en la carpeta `pruebas/`
2. Lista todos los archivos encontrados
3. Procesa cada archivo secuencialmente
4. Muestra predicción y visualización para cada uno

### Formatos Soportados
- ✅ WAV (nativo)
- ⚠️ M4A (requiere conversión)
- ⚠️ MP3 (requiere conversión)
- ⚠️ FLAC (requiere conversión)
- ⚠️ OGG (requiere conversión)

---

## 📈 Resultados del Modelo

### Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| Precisión en Entrenamiento | 98.25% |
| Precisión en Validación | 100% |
| Precisión en Prueba | 100% |
| Pérdida Final | 0.0015 |
| Tiempo de Entrenamiento | ~5 minutos |

### Matriz de Confusión
El modelo alcanza **100% de precisión** en el conjunto de prueba, lo que significa:
- ✅ Cero falsos positivos
- ✅ Cero falsos negativos
- ✅ Clasificación perfecta para todas las clases

### Curvas de Aprendizaje
- La precisión de validación alcanza 100% en la época 48
- La pérdida de validación converge a ~0.0015
- No se observa overfitting gracias al dropout

---

## 🎓 Conceptos Técnicos

### MFCC (Mel-Frequency Cepstral Coefficients)
Los MFCC son características que representan el espectro de potencia a corto plazo de un sonido, basándose en una transformación de coseno lineal de un espectro de potencia logarítmica en una escala de frecuencia mel no lineal.

**¿Por qué MFCC?**
- Imita la percepción auditiva humana
- Reduce la dimensionalidad del audio
- Captura características fonéticas importantes
- Robusto ante variaciones de tono

### Red Neuronal Multicapa (MLP)
Una MLP es una red neuronal feedforward que consiste en al menos tres capas de nodos: una capa de entrada, una o más capas ocultas y una capa de salida.

**Ventajas para clasificación de audio:**
- Aprende representaciones no lineales
- Maneja datos de alta dimensionalidad
- Generaliza bien con suficiente regularización

### Dropout
Técnica de regularización que desactiva aleatoriamente neuronas durante el entrenamiento para prevenir overfitting.

**En este modelo:**
- 30% en las primeras capas
- 20% en las capas intermedias
- Mejora la generalización

---

## 🚀 Uso Práctico

### Entrenar el Modelo

```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Ejecutar entrenamiento
python app.py
```

### Probar con Nuevo Audio

1. **Grabar o obtener un audio**
   - Di un número del 0 al 9
   - Guarda como archivo de audio

2. **Convertir a WAV (si es necesario)**
   - Usar herramienta online: https://convertio.co/es/m4a-wav/
   - O con VLC: Media > Convert/Save > Audio - CD

3. **Colocar en carpeta pruebas**
```powershell
# Copiar archivo
Copy-Item "ruta/al/audio.wav" "pruebas/"
```

4. **Ejecutar predicción**
```powershell
python test_audio.py
```

5. **Ver resultados**
   - Terminal: Número predicho y confianza
   - Ventana emergente: Visualizaciones gráficas

---

## 🔧 Solución de Problemas

### Error: "No module named 'tensorflow'"
```powershell
pip install tensorflow librosa matplotlib scikit-learn
```

### Error: "No se encontró el dataset"
- Verifica que la carpeta `content/digit_dataset/` existe
- Asegúrate de tener archivos WAV en el dataset
- Revisa la ruta en `app.py` línea 188

### Error: "Format not recognised" (archivos M4A)
- Convierte el audio a WAV antes de procesar
- Usa: https://convertio.co/es/m4a-wav/
- O instala FFmpeg y configura el PATH

### Precisión baja en tus audios
- Asegúrate de que el audio sea claro
- Verifica que solo contenga el número (sin ruido)
- Comprueba que la duración sea similar al dataset (~1 segundo)
- Prueba con diferentes personas/acentos

---

## 📝 Archivos Auxiliares

### create_test_audio.py
Genera un archivo WAV sintético para pruebas rápidas sin necesidad de grabar.

```powershell
python create_test_audio.py
```

### convert_m4a_to_wav.py
Muestra instrucciones detalladas para convertir archivos M4A a WAV.

```powershell
python convert_m4a_to_wav.py
```

---

## 🎯 Casos de Uso

1. **Sistemas de respuesta por voz (IVR)**
   - Menús telefónicos automatizados
   - Navegación por comandos de voz

2. **Accesibilidad**
   - Entrada de datos por voz para personas con discapacidad
   - Control de dispositivos mediante voz

3. **Educación**
   - Aplicaciones de aprendizaje de números
   - Evaluación automática de pronunciación

4. **Domótica**
   - Control de dispositivos con comandos numéricos
   - Sistemas de seguridad con código PIN por voz

---

## 👨‍💻 Desarrollo

**Proyecto desarrollado como parte de:**
- Curso: Inteligencia Artificial
- UCEVA: ING SISTEMAS - Semestre 7
- Fecha: Noviembre 2025

### Tecnologías Implementadas:
- ✅ Deep Learning (MLP)
- ✅ Procesamiento de Señales de Audio
- ✅ Extracción de Características (MFCC)
- ✅ Visualización de Datos
- ✅ Regularización (Dropout)
- ✅ Callbacks de Keras (EarlyStopping, ReduceLROnPlateau)

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

---
- Presentado por:
- Laura Sofia Toro Garcia
- Santiago Martinez Serna
- Santiago Alejandro Santacruz

