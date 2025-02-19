# **Adam Implementation - Machine Learning**

This project implements the **Adam optimization algorithm** to train a machine learning model for classifying wine quality based on physicochemical properties. The implementation is available in both **Python** and **Node.js**.

---
## **📌 Project Structure**
```
adam-implementation/
│── javascript/
│   ├── index.js  # Node.js implementation
│
│── python/
│   ├── env/  # Virtual environment for Python dependencies
│   ├── wine_quality_ml.py  # Python implementation
│   ├── requirements.txt  # List of Python dependencies
```
---
## **🔹 Running the Project in Python**

### **1️⃣ Prerequisites**
Ensure you have **Python 3.x** installed. If not, download it from [Python's official website](https://www.python.org/downloads/).

### **2️⃣ Activate the Virtual Environment**
Navigate to the `python` folder and activate the existing virtual environment:

#### **Windows**
```bash
cd python
env\Scripts\activate
```

#### **Mac/Linux**
```bash
cd python
source env/bin/activate
```

### **3️⃣ Install `pip` (if missing)**
If `pip` is not installed, run:
```bash
python -m ensurepip --default-pip
```
Then, update it to the latest version:
```bash
pip install --upgrade pip
```

### **4️⃣ Install Dependencies**
If `requirements.txt` does not exist, create it with the following:
```bash
pip freeze > requirements.txt
```
Then, install the required packages:
```bash
pip install -r requirements.txt
```

### **5️⃣ Run the Python Script**
Execute the script to train and evaluate the model:
```bash
python wine_quality_ml.py
```

### **6️⃣ Expected Output**
The script will:
- Load and preprocess the dataset.
- Train the model using **Adam optimizer**.
- Evaluate the model performance.
- Display accuracy and loss metrics.

---
## **🔹 Running the Project in Node.js**

### **1️⃣ Prerequisites**
Ensure you have **Node.js** and **npm** installed. If not, download them from [Node.js official website](https://nodejs.org/).

### **2️⃣ Install Dependencies**
Navigate to the `javascript` folder and install the required packages:
```bash
cd javascript
npm install
```

### **3️⃣ Run the Node.js Script**
Execute the script to train and evaluate the model:
```bash
node index.js
```

### **4️⃣ Expected Output**
The script will:
- Load and preprocess the dataset.
- Train the model using **Adam optimizer**.
- Compare results with **Stochastic Gradient Descent (SGD)**.
- Display accuracy and loss metrics.

---
## **🚀 Next Steps**
- Fine-tune hyperparameters for better performance.
- Add visualization of loss and accuracy trends.
- Experiment with different optimizers like **RMSProp** or **Momentum**.

For any issues or contributions, feel free to open a pull request. Happy coding! 🚀

