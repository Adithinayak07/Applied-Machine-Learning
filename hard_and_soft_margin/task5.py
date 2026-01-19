from sklearn.svm import SVC
from data import overlap_data
from visual import plot_soft_margin_svm

def main():
    # Load overlapping data
    X_overlap, y_overlap = overlap_data()

    # Soft margin SVM
    svm_soft = SVC(kernel='linear', C=1.0)
    svm_soft.fit(X_overlap, y_overlap)

    print("Number of support vectors:", len(svm_soft.support_vectors_))
    print("w:", svm_soft.coef_[0])
    print("b:", svm_soft.intercept_[0])

    # Plot results
    plot_soft_margin_svm(X_overlap, y_overlap, svm_soft)

if __name__ == "__main__":
    main()
