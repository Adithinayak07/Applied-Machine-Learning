import matplotlib.pyplot as plt 
import numpy as np

def plot_2d_data(X,y, title="Linear Data"):
    plt.scatter(X[:,0],X[:,1],c=y, cmap='bwr')
    plt.title("Linearly Separable Data")
    plt.show()

def plot_hard_margin_svm(X,y,svm_hard):
    w = svm_hard.coef_[0] #weight vector
    b = svm_hard.intercept_[0]

    plt.figure(figsize=(7,6))
    # Plot data points
    plt.scatter(X[:,0],X[:,1], c=y, cmap='bwr', edgecolors ='k')

    #Highlight support vectors
    plt.scatter(
        svm_hard.support_vectors_[:,0],
        svm_hard.support_vectors_[:,1],
        s = 120, facecolors='None', edgecolors='k',linewidth=2,
        label = "Support Vectors"
    )

    # Create x values for line plotting
    x_vals = np.linspace(X[:,0].min()-1,X[:,0].max()+1,200)

    # Decision boundry: w-x + b =0
    y_decision = -(w[0] * x_vals +b)/w[1]

    # Margin boundaries: w.x + b = +_1
    y_margin_pos = -(w[0] * x_vals +b-1)/w[1]
    y_margin_neg = -(w[0] * x_vals +b+1)/w[1]

    #plot lines
    plt.plot(x_vals, y_decision,'k-',label="Decision Boundry")
    plt.plot(x_vals, y_margin_pos,'k--',label="Margin +1")
    plt.plot(x_vals, y_margin_neg,'k--',label="Margin -1")

    # Shade margin area 
    plt.fill_between(
        x_vals, y_margin_pos,y_margin_neg,
        color='gray',alpha =0.2, label = "Margin Area"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Hard Margin SVM with Margin Visualization")
    plt.legend()
    plt.show()


def plot_overlapping_data(X_overlap, y_overlap):
    plt.scatter(X_overlap[:,0],X_overlap[:,1],c=y_overlap,cmap = 'bwr')
    plt.title("Overlapping Data")
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_svm_overlap(X_overlap, y_overlap, svm_model):
    w = svm_model.coef_[0]
    b = svm_model.intercept_[0]

    plt.figure(figsize=(7, 6))

    # Plot points
    plt.scatter(
        X_overlap[:, 0], X_overlap[:, 1],
        c=y_overlap, cmap='bwr', edgecolors='k'
    )

    # Support vectors
    plt.scatter(
        svm_model.support_vectors_[:, 0],
        svm_model.support_vectors_[:, 1],
        s=120, facecolors='None', edgecolors='k',
        linewidths=2, label="Support Vectors"
    )

    # x values
    x_vals = np.linspace(
        X_overlap[:, 0].min() - 1,
        X_overlap[:, 0].max() + 1,
        200
    )

    # Decision boundary and margins
    y_decision = -(w[0] * x_vals + b) / w[1]
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]

    # Plot lines
    plt.plot(x_vals, y_decision, 'k-', label="Decision Boundary")
    plt.plot(x_vals, y_margin_pos, 'k--', label="Margin +1")
    plt.plot(x_vals, y_margin_neg, 'k--', label="Margin -1")

    # Shade margin area
    plt.fill_between(
        x_vals, y_margin_pos, y_margin_neg,
        color='gray', alpha=0.2, label="Margin Area"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Hard Margin SVM on Overlapping Data")
    plt.legend()
    plt.show()



def plot_soft_margin_svm(X_overlap, y_overlap, svm_model, title="Soft Margin SVM (C=1.0)"):
    w_soft = svm_model.coef_[0]
    b_soft = svm_model.intercept_[0]

    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_overlap[:, 0], X_overlap[:, 1],
        c=y_overlap, cmap='bwr', edgecolors='k'
    )

    # Support vectors
    plt.scatter(
        svm_model.support_vectors_[:, 0],
        svm_model.support_vectors_[:, 1],
        s=120, facecolors='None', edgecolors='k',
        linewidth=2, label="Support Vectors"
    )

    # Decision boundary and margins
    x_vals = np.linspace(
        X_overlap[:, 0].min() - 1,
        X_overlap[:, 0].max() + 1,
        200
    )

    y_decision = -(w_soft[0] * x_vals + b_soft) / w_soft[1]
    y_margin_pos = -(w_soft[0] * x_vals + b_soft - 1) / w_soft[1]
    y_margin_neg = -(w_soft[0] * x_vals + b_soft + 1) / w_soft[1]

    plt.plot(x_vals, y_decision, 'k-', label="Decision Boundary")
    plt.plot(x_vals, y_margin_pos, 'k--', label="Margin +1")
    plt.plot(x_vals, y_margin_neg, 'k--', label="Margin -1")

    # Shade margin area
    plt.fill_between(
        x_vals, y_margin_pos, y_margin_neg,
        color='gray', alpha=0.2, label="Margin Area"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()



def plot_svm(X, y, svm_model, title="SVM"):
    w = svm_model.coef_[0]
    b = svm_model.intercept_[0]

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    # Support vectors
    plt.scatter(
        svm_model.support_vectors_[:, 0],
        svm_model.support_vectors_[:, 1],
        s=120, facecolors='None', edgecolors='k',
        linewidth=2, label="Support Vectors"
    )

    # Decision boundary and margins
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)

    y_decision = -(w[0] * x_vals + b) / w[1]
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]

    plt.plot(x_vals, y_decision, 'k-', label="Decision Boundary")
    plt.plot(x_vals, y_margin_pos, 'k--', label="Margin +1")
    plt.plot(x_vals, y_margin_neg, 'k--', label="Margin -1")

    plt.fill_between(
        x_vals, y_margin_pos, y_margin_neg,
        color='gray', alpha=0.2, label="Margin Area"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()
