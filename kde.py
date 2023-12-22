from scipy import stats


class kernel_density():
    def __init__(self):
        pass
    def compute(self,dim,data, X_new):

        values = [[] for _ in range(dim)]
        for i in range(len(data)):
            for d in range(dim):
                values[d].append(data[i][d])
        kernel = stats.gaussian_kde(values)
        X_test=[[] for _ in range(dim)]

        for i in range( len(X_new)):
            for d in range(dim):
                X_test[d].append(X_new[i][d])
        return kernel(X_test)