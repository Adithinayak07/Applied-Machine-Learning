from data import overlap_data
from visual import plot_overlapping_data

def main():
    X_overlap, y_overlap= overlap_data()
    
    plot_overlapping_data(X_overlap,y_overlap)
if __name__ == "__main__":
    main()