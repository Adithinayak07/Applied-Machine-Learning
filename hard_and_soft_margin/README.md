# 📌 SVM Decision Boundary & Margin Visualization (Hard vs Soft Margin)

This project demonstrates **Support Vector Machines (SVM)** using `scikit-learn` and shows how the **decision boundary, margins, and support vectors** change for:

✅ Linearly separable data  
✅ Overlapping data  
✅ Hard Margin SVM (very large C)  
✅ Soft Margin SVM (different values of C)

---

## 🎯 Project Objectives

- Generate synthetic classification datasets using `make_blobs`
- Train **Linear SVM**
- Visualize:
  - Decision boundary
  - Margin lines (+1 and -1)
  - Margin area
  - Support vectors
- Study the effect of **C parameter** in soft-margin SVM

---

## 🧠 Concepts Covered

- Linearly separable vs overlapping data
- Hard Margin SVM vs Soft Margin SVM
- Support vectors and their role in classification
- Effect of regularization parameter **C**
  - Small C → wider margin, allows misclassification
  - Large C → narrower margin, less misclassification
  - Very large C → behaves like hard margin

---

## 📂 Project Structure

```bash
hard_and_soft_margin/
│
├── data.py          # Dataset generation functions
├── visual.py        # Visualization functions
│
├── task1.py         # Task 1: Linearly separable data generation + plot
├── task2.py         # Task 2: Hard margin SVM on separable data + plot
├── task3.py         # Task 3: Overlapping data generation + plot
├── task4.py         # Task 4: Hard margin SVM on overlapping data + plot
├── task5.py         # Task 5: Soft margin SVM (C=1.0) on overlapping data + plot
├── task6.py         # Task 6: Soft margin SVM for multiple C values + plot
│
└── README.md        # Documentation
````

---

## ✅ Tasks Explanation

---

### ✅ Task 1: Generate Linearly Separable Data

📌 File: `task1.py`

* Generates data using `make_blobs`
* Converts class labels `{0,1}` → `{-1,1}`
* Visualizes scatter plot

---

### ✅ Task 2: Hard Margin SVM on Linearly Separable Data

📌 File: `task2.py`

* Trains **Hard Margin SVM** with very large `C = 1e6`
* Prints:

  * Support vectors count
  * Weight vector `w`
  * Bias `b`
* Plots:

  * decision boundary
  * margins
  * support vectors

---

### ✅ Task 3: Generate Overlapping Data

📌 File: `task3.py`

* Generates overlapping dataset with larger standard deviation `cluster_std = 3`
* Visualizes scatter plot

---

### ✅ Task 4: Hard Margin SVM on Overlapping Data

📌 File: `task4.py`

* Trains **Hard Margin SVM** on overlapping data
* Shows how hard margin struggles when data is not perfectly separable

---

### ✅ Task 5: Soft Margin SVM on Overlapping Data (C = 1.0)

📌 File: `task5.py`

* Trains **Soft Margin SVM**
* Uses `C=1.0`
* Produces better generalization by allowing some margin violations

---

### ✅ Task 6: Soft Margin SVM for Different C Values

📌 File: `task6.py`

Trains Soft Margin SVM for multiple values:

```python
C = [10, 100, 1000]
```

Observation:

* for large C values, decision boundary becomes stable
* support vectors may remain same (depends on dataset)

---

## ▶️ How to Run

### ✅ Install requirements

```bash
pip install -r requirements.txt
```

### ✅ Run any task

```bash
python task1.py
python task2.py
python task3.py
python task4.py
python task5.py
python task6.py
```

---

## 📌 Output Examples

Each task will generate plots showing:

✅ Data distribution
✅ Decision boundary
✅ Margin lines
✅ Support vectors
✅ Margin shaded region

---

## 🏁 Conclusion

This project clearly illustrates:

* How SVM separates data using maximum margin principle
* Why support vectors are critical
* How **C** controls the trade-off between:

  * larger margin and misclassification
  * smaller margin and strict classification

---

## 👩‍💻 Author

**Adithi Nayak**
SVM Practical Implementation & Visualization Project


