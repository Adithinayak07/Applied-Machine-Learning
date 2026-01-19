from sklearn.svm import SVC
from data import generate_linear_data
from visual import plot_hard_margin_svm

def main():
    X, y= generate_linear_data()
    svm_hard = SVC(kernel = 'linear',C = 1e6)
    svm_hard.fit(X,y)

    print("Number of support vectors: ", len(svm_hard.support_vectors_))
   

    print("w:",svm_hard.coef_[0])
    print("b:",svm_hard.intercept_[0])
    plot_hard_margin_svm(X,y,svm_hard)
if __name__ == "__main__":
    main()