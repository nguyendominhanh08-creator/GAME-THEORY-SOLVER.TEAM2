import numpy as np

def find_saddle_points(A):
    """
    Tìm tất cả các điểm yên ngựa trong ma trận A.
    Trả về danh sách các tuple (row_index, col_index).
    """
    A = np.asarray(A, dtype=float)
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    
    maximin = np.max(row_mins)
    minimax = np.min(col_maxs)
    
    saddle_points = []
    if np.isclose(maximin, minimax):
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isclose(A[i, j], maximin) and np.isclose(row_mins[i], maximin) and np.isclose(col_maxs[j], minimax):
                    saddle_points.append((int(i), int(j)))
                    
    return saddle_points
