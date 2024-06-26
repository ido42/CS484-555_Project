import numpy as np

# classification task not regression
class Weaker:
    def fit(self, X, y, w):
        n_samples, n_features = X.shape

        self.__n_subsets = int(np.sqrt(n_samples))
        z_min = np.inf
        for i in range(n_features):
            w_positive = np.zeros(self.__n_subsets)
            w_negative = np.zeros(self.__n_subsets)

            split_points = np.linspace(min(X[:, i]), max(X[:, i]), self.__n_subsets + 1)
            split_points[0] = -np.inf
            split_points[self.__n_subsets] = np.inf
            for k in range(self.__n_subsets):
                subset_indexes = np.flatnonzero((split_points[k] < X[:, i]) & (X[:, i] <= split_points[k + 1]))
                w_positive[k] = sum(w[subset_indexes][np.flatnonzero(y[subset_indexes] == 1)])
                w_negative[k] = sum(w[subset_indexes][np.flatnonzero(y[subset_indexes] == -1)])

            h = (w_positive - w_negative) / (w_positive + w_negative + 1e-8)

            out = np.zeros(n_samples)
            for j in range(n_samples):
                for k in range(self.__n_subsets):
                    if split_points[k] < X[j, i] <= split_points[k + 1]:
                        out[j] = h[k]
                        break
            z = np.sum(w * (out - y) ** 2)

            if z < z_min:
                z_min = z
                self.__feature = i
                self.__split_points = split_points
                self.__h = h

    def predict(self, X):
        n_samples = X.shape[0]

        h = np.zeros(n_samples)
        for i in range(n_samples):
            for k in range(self.__n_subsets):
                if self.__split_points[k] < X[i, self.__feature] <= self.__split_points[k + 1]:
                    h[i] = self.__h[k]
                    break

        return h


class GentleAdaboost:
    def fit(self, X, y, n_estimators):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values, 1 or -1
        n_estimators : The number of estimators at which boosting is terminated
        '''
        n_samples, n_features = X.shape

        self.__estimators = []

        w = np.full(n_samples, 1 / n_samples)
        for i in range(n_estimators):
            model = Weaker()
            model.fit(X, y, w)
            h = model.predict(X)

            w *= np.exp(-y * h)

            self.__estimators.append(model)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample, 1 or -1
        '''
        h = self.score(X)

        y_pred = np.ones_like(h)
        y_pred[np.flatnonzero(h < 0)] = -1

        return y_pred

    def score(self, X):
        return sum([classifier.predict(X) for classifier in self.__estimators])