import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor


class TrajectoryPrediction:

    """
    Class to predict a path based on given points.
    """

    def __init__(self, min_samples: int):
        """
        Creates a TrajectoryPrediction object.

        """
        self.Xr = RANSACRegressor(min_samples=min_samples)
        self.Yr = PolynomialFeatures()
        self.YrR = RANSACRegressor(min_samples=min_samples)
        self.Zr = RANSACRegressor(min_samples=min_samples)

    def predict_path(self, path: np.array, next_points: int) -> np.array:

        """
            Predicts the next next_points points of the ball's path.

        :param path: Numpy Array of all positions
        :param next_points: Number of next points to predict
        :return: Numpy Array of predicted next points
        """

        ts = np.arange(path.shape[0])[:, np.newaxis]
        xr = self.Xr.fit(ts, path[:, 0]) # Lineare Regression über die X Werte
        ts_transformed = self.Yr.fit_transform(ts) # polynomische Features für die Zeit

        yr = self.YrR.fit(ts_transformed, path[:, 1])
        zr = self.Zr.fit(ts, path[:, 2]) # finde zu den zugehörigen x-Werten (ts) die y-Werte (path[:, 2])

        ts = np.arange(path.shape[0] + next_points)[:, np.newaxis]
        Y_transformed = self.Yr.fit_transform(np.arange(ts.shape[0])[:, np.newaxis])

        pred_path = np.zeros(ts.shape[0] * 3).reshape(-1, 3)
        pred_path[:, 0] = xr.predict(np.arange(ts.shape[0])[:, np.newaxis])
        pred_path[:, 1] = yr.predict(Y_transformed)
        pred_path[:, 2] = zr.predict(np.arange(ts.shape[0])[:, np.newaxis])
        return pred_path
