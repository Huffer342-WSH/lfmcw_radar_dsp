# %%
import numpy as np
from scipy.optimize import linear_sum_assignment


# %%
def GNN_findMinCost(cost: np.ndarray):
    m, n = cost.shape
    indices = np.arange(n)
    cycles = np.arange(n, n - m, -1)
    best_indices = indices[:m].copy()
    best_cost = np.sum(cost[np.arange(m), best_indices])
    while n:
        flag = True
        for i in range(m - 1, -1, -1):
            cycles[i] -= 1
            if cycles[i] == 0:
                temp = indices[i]
                for j in range(i, n - 1):
                    indices[j] = indices[j + 1]
                indices[n - 1] = temp
                cycles[i] = n - i
            else:
                j = cycles[i]
                temp = indices[n - j]
                indices[n - j] = indices[i]
                indices[i] = temp
                if best_cost >= np.sum(cost[np.arange(m), indices[:m]]):
                    best_indices = indices[:m].copy()
                    best_cost = np.sum(cost[np.arange(m), best_indices])
                flag = False
                break
        if flag:
            break
    return best_indices, best_cost


a = np.random.randn(35).reshape(5, 7)
indices, cost = GNN_findMinCost(a)
row_ind, col_ind = linear_sum_assignment(a)
assert np.sum(a[row_ind, col_ind]) == cost
print(a)
print(indices, col_ind)
print(cost)


# %%
