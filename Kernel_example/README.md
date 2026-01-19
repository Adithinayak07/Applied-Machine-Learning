## ✅ `README.md` (for Kernel folder)

````md
# 🧠 Kernel Methods: Linear vs Non-Linear Classification (XOR Problem)

This mini-project demonstrates why **linear models fail on non-linear datasets** like XOR, and how **feature transformation** and **kernel methods** (Polynomial, RBF) help solve the problem.

The project uses synthetic data and visualizations to compare:
✅ Linearly separable data  
✅ XOR (non-linearly separable) data  
✅ Linear models vs Kernel-based models  

---

## 🎯 Objectives

- Generate and visualize:
  - **Linearly separable dataset**
  - **XOR dataset (non-linearly separable)**
- Show that:
  - Linear classifiers fail on XOR
  - Feature transformations (polynomial features) help
  - SVM kernels can perfectly solve XOR

---

## 📂 Folder Structure

```bash
Kernel_example/
│
├── data.py               # Data generation functions
├── visual.py             # Plotting functions
│
├── task0.py              # Linearly separable data visualization
├── task1.py              # XOR data visualization
├── task2.py              # Logistic Regression on XOR (before & after polynomial features)
├── task3.py              # SVM kernels on XOR (Linear, Polynomial, RBF)
│
├── requirements.txt      # Libraries required
└── README.md             # Documentation
````

---

## 🧪 Tasks Explanation

---

### ✅ Task 0: Linearly Separable Data

📌 `task0.py`

* Generates linearly separable dataset
* Displays scatter plot

✅ Expected output: clear separation possible using a straight line.

---

### ✅ Task 1: XOR (Non-Linearly Separable) Data

📌 `task1.py`

* Generates XOR dataset
* Displays scatter plot

✅ Expected output: classes form a pattern that cannot be separated using a straight line.

---

### ✅ Task 2: Logistic Regression on XOR (Feature Transformation)

📌 `task2.py`

This task shows two experiments:

#### 1️⃣ Logistic Regression in Original Space

* Trains logistic regression directly on XOR data
* Accuracy is low since XOR is non-linear

#### 2️⃣ Logistic Regression After Polynomial Transformation

* Converts features using `PolynomialFeatures(degree=2)`
* Then applies logistic regression
* Accuracy improves significantly

✅ Key Learning:

> Feature transformation makes XOR linearly separable in higher-dimensional space.

---

### ✅ Task 3: SVM Kernel Methods on XOR

📌 `task3.py`

Trains 3 SVM models on XOR:

* **Linear Kernel SVM** ❌ (fails)
* **Polynomial Kernel SVM** ✅ (works)
* **RBF Kernel SVM** ✅ (works)

✅ Key Learning:

> Kernel trick allows SVM to classify XOR without explicitly transforming features.

---

## 📊 Kernels Covered

| Kernel     | Works on XOR? | Reason                                |
| ---------- | ------------- | ------------------------------------- |
| Linear     | ❌ No          | XOR is not linearly separable         |
| Polynomial | ✅ Yes         | Maps data to higher dimension         |
| RBF        | ✅ Yes         | Non-linear flexible decision boundary |

---

## ▶️ How to Run

### ✅ Install requirements

```bash
pip install -r requirements.txt
```

### ✅ Run tasks

```bash
python task0.py
python task1.py
python task2.py
python task3.py
```

Each task generates scatter plots and prints relevant accuracy outputs.

---

## 🏁 Conclusion

This project demonstrates:

* Difference between **linear vs non-linear separability**
* Importance of **feature transformation**
* Power of **SVM kernels** for solving complex datasets like XOR

---

## 👩‍💻 Author

**Adithi Nayak**
Kernel Methods & XOR Classification - Applied Machine Learning Lab

````

---
