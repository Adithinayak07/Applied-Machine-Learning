# 🧬 Breast Cancer Classification using SVM & PCA

This project demonstrates **data visualization and classification** on the **Breast Cancer dataset** using **Support Vector Machines (SVM)** and **Principal Component Analysis (PCA)**.

The goal is to:
✅ Visualize high-dimensional medical data in 2D  
✅ Compare SVM kernels (Linear, Polynomial, RBF)  
✅ Observe model behavior on original vs extended datasets  

---

## 🎯 Objectives

- Load and explore the Breast Cancer dataset
- Visualize data in **2D using PCA**
- Train SVM classifiers with different kernels
- Compare accuracy across kernels
- Understand the effect of dataset scaling via resampling

---

## 📂 Folder Structure

```bash
Breast_Cancer_SVM/
│
├── data.py          # Dataset loading, resampling, PCA transformations
├── visual.py        # 2D visualization utility
│
├── task0.py         # Dataset visualization (original & extended)
├── task1.py         # SVM training and evaluation
│
└── README.md        # Documentation
```

---

## 📊 Dataset Description

- **Source**: `sklearn.datasets.load_breast_cancer`
- **Classes**:
  - `0` → Malignant
  - `1` → Benign
- **Original Features**: 30 numerical features
- **Type**: Binary classification (medical dataset)

---

## 🧪 Tasks Explanation

---

### ✅ Task 0: Dataset Visualization

📌 `task0.py`

This task focuses on **understanding and visualizing the data**.

#### 1️⃣ Original Dataset
- Loads breast cancer data
- Visualizes it directly in 2D

#### 2️⃣ Extended Dataset
- Increases number of samples using **resampling**
- Reduces dimensionality using **PCA**
- Visualizes the PCA-reduced dataset

✅ Key Learning:

> PCA helps visualize high-dimensional data, but does not change class labels.

Run:
```bash
python task0.py
```

---

### ✅ Task 1: SVM Classification

📌 `task1.py`

This task trains **Support Vector Machine classifiers** using different kernels.

Models trained:
- **Linear Kernel SVM**
- **Polynomial Kernel SVM (degree = 2)**
- **RBF Kernel SVM**

For each model:
- Predictions are visualized in 2D
- Training accuracy is printed

By default, the **extended dataset** is used.

Run:
```bash
python task1.py
```

> To run on original data, uncomment `original_data()` in `task1.py`.

---

## 🧠 Machine Learning Concepts Covered

- Support Vector Machines (SVM)
- Kernel methods:
  - Linear
  - Polynomial
  - Radial Basis Function (RBF)
- Principal Component Analysis (PCA)
- Bootstrap Resampling
- Classification accuracy evaluation

---

## 📊 Kernel Comparison

| Kernel     | Performance | Observation |
|-----------|-------------|-------------|
| Linear    | Moderate    | Works well if data is close to linear |
| Polynomial | Better     | Captures feature interactions |
| RBF       | Best       | Flexible non-linear decision boundary |

---

## ▶️ How to Run

### ✅ Install dependencies

```bash
pip install numpy matplotlib scikit-learn
```

### ✅ Execute tasks

```bash
python task0.py
python task1.py
```

Each script generates visual plots and prints accuracy values.

---

## ⚠️ Notes

- Accuracy is calculated on **training data only**
- PCA is used **for visualization purposes**
- Large sample sizes may increase runtime

---

## 🏁 Conclusion

This project demonstrates:

- How SVM kernels behave on real medical data
- The importance of dimensionality reduction for visualization
- Practical application of PCA and kernel-based classifiers

It serves as a **foundational machine learning experiment** combining theory with hands-on implementation.

---

## 👩‍💻 Author

**Adithi Nayak**  
Breast Cancer Classification using SVM & PCA  
Applied Machine Learning Lab

