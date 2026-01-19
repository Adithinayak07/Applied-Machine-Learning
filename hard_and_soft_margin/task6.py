from sklearn.svm import SVC
from data import overlap_data
from visual import plot_svm

def main():
    # Get overlapping dataset
    X_overlap, y_overlap = overlap_data()

    # Different C values
    C_values = [0.01, 0.1, 1, 10]

    for C in C_values:
        svm_soft = SVC(kernel="linear", C=C)
        svm_soft.fit(X_overlap, y_overlap)

        print(f"\n===== Soft Margin SVM (C={C}) =====")
        print("Number of support vectors:", len(svm_soft.support_vectors_))
        print("w:", svm_soft.coef_[0])
        print("b:", svm_soft.intercept_[0])

        plot_svm(X_overlap, y_overlap, svm_soft, title=f"Soft Margin SVM (C={C})")

if __name__ == "__main__":
    main()
