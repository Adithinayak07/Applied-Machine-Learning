from sklearn.svm import SVC
from data import overlap_data
from visual import plot_svm_overlap

def main():
    # Get overlapping dataset
    X_overlap, y_overlap = overlap_data()

    # Hard Margin SVM (very large C)
    svm_hard = SVC(kernel='linear', C=1e6)
    svm_hard.fit(X_overlap, y_overlap)

    print("Number of support vectors:", len(svm_hard.support_vectors_))
    print("w:", svm_hard.coef_[0])
    print("b:", svm_hard.intercept_[0])

    # Plot decision boundary
    plot_svm_overlap(X_overlap, y_overlap, svm_hard)

if __name__ == "__main__":
    main()
