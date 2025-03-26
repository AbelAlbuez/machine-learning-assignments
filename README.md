# **Proyecto de Machine Learning**

Este repositorio contiene dos proyectos relacionados con machine learning:

1. **Implementación del algoritmo de optimización Adam** para clasificación de calidad de vinos
2. **Redes Neuronales Feedforward para MNIST** (Taller 2)

---

## **📁 Estructura del Proyecto**

```
machine-learning-assignments/
│
├── adam-implementation/
│   ├── datasets/
│   ├── javascript/
│   │   └── index.js  # Implementación en Node.js
│   ├── python/
│   │   ├── env/  # Entorno virtual para dependencias Python
│   │   ├── wine_quality_ml.py  # Implementación en Python
│   │   └── requirements.txt  # Lista de dependencias Python
│   └── Bitácora de Desarrollo - Implementación del Algoritmo Adam.pdf
│
└── Taller 2/
    ├── data/
    ├── examples/
    ├── lib/
    │   └── PUJ_ML/
    ├── MNIST_ORG/
    ├── models/
    ├── venv/
    ├── Taller 2 - Bitácora de Desarrollo.docx
    └── README.md
```

---

## **💻 Implementación de Adam**

### **🔹 Ejecutando el Proyecto en Python**

#### **1️⃣ Prerrequisitos**

Asegúrate de tener **Python 3.x** instalado. Si no, descárgalo desde [el sitio oficial de Python](https://www.python.org/downloads/).

#### **2️⃣ Activar el Entorno Virtual**

Navega a la carpeta `adam-implementation/python` y activa el entorno virtual existente:

##### **Windows**

```bash
cd adam-implementation/python
env\Scripts\activate
```

##### **Mac/Linux**

```bash
cd adam-implementation/python
source env/bin/activate
```

#### **3️⃣ Instalar Dependencias**

Instala los paquetes requeridos:

```bash
pip install -r requirements.txt
```

#### **4️⃣ Ejecutar el Script Python**

```bash
python wine_quality_ml.py
```

### **🔹 Ejecutando el Proyecto en Node.js**

#### **1️⃣ Prerrequisitos**

Asegúrate de tener **Node.js** y **npm** instalados. Si no, descárgalos desde [el sitio oficial de Node.js](https://nodejs.org/).

#### **2️⃣ Instalar Dependencias**

Navega a la carpeta `adam-implementation/javascript` e instala los paquetes requeridos:

```bash
cd adam-implementation/javascript
npm install
```

#### **3️⃣ Ejecutar el Script Node.js**

```bash
node index.js
```

---

## **🧠 Taller 2: Redes Neuronales para MNIST**

### **📋 Descripción**

Este proyecto implementa tres modelos diferentes de redes neuronales feedforward para la clasificación de dígitos manuscritos del dataset MNIST.

### **🔹 Ejecutando el Proyecto**

#### **1️⃣ Prerrequisitos**

Asegúrate de tener **Python 3.x** instalado con las siguientes bibliotecas:

- NumPy
- Matplotlib
- scikit-learn

#### **2️⃣ Activar el Entorno Virtual**

Navega a la carpeta `Taller 2` y activa el entorno virtual:

##### **Windows**

```bash
cd "Taller 2"
venv\Scripts\activate
```

##### **Mac/Linux**

```bash
cd "Taller 2"
source venv/bin/activate
```

#### **3️⃣ Instalar Dependencias**

Si las dependencias no están instaladas:

```bash
pip install numpy matplotlib scikit-learn
```

#### **4️⃣ Ejecutar el Script Principal**

```bash
python main.py
```

### **🔢 Modelos Implementados**

1. **Red Neuronal Simple**
   - Arquitectura: 784 → 100 (Sigmoid) → 10 (Softmax)
   - Inicialización: Xavier

2. **Red Neuronal Profunda**
   - Arquitectura: 784 → 100 (ReLU) → 50 (ReLU) → 10 (Softmax)
   - Inicialización: He

3. **Red Neuronal Personalizada**
   - Arquitectura: 784 → 30 (Sigmoid) → 30 (Sigmoid) → 10 (Softmax)
   - Inicialización: Xavier

### **🛠️ Mejoras Implementadas**

- **Optimización Softmax**: Implementación estable con manejo de batches grandes
- **Inicialización de Pesos**: Métodos Xavier y He
- **Optimización de Adam**: Mejoras para manejo eficiente de memoria
- **Análisis de Resultados**: Visualización detallada de matrices de confusión y curvas de aprendizaje

---

## **📊 Resultados**

Los resultados y métricas de cada modelo se pueden encontrar en:

- `adam-implementation/Bitácora de Desarrollo - Implementación del Algoritmo Adam.pdf`
- `Taller 2/Taller 2 - Bitácora de Desarrollo.docx`

---
