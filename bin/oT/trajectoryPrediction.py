import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures


class TrajectoryPrediction:
    """
    Class to predict a path based on given points.
    """

    def __init__(self, min_samples: int):
        """
        Creates a TrajectoryPrediction object.

        """
        self.X_ransac = RANSACRegressor(min_samples=min_samples)
        self.t_poly = PolynomialFeatures()
        self.Y_ransac = RANSACRegressor(min_samples=min_samples)
        self.Z_ransac = RANSACRegressor(min_samples=min_samples)

    def predict_path(self, path: np.array, next_points: int) -> np.array:
        """
            Predicts the next next_points points of the ball's path.

        :param path: Numpy Array of all positions
        :param next_points: Number of next points to predict
        :return: Numpy Array of predicted next points
        """
        # extract time and create polynomial features from it
        time = np.arange(path.shape[0])[:, np.newaxis]
        time_transformed = self.t_poly.fit_transform(time)

        # create RANSAC regression model for X, Y and Z
        xr = self.X_ransac.fit(time, path[:, 0])
        yr = self.Y_ransac.fit(time_transformed, path[:, 1])
        zr = self.Z_ransac.fit(time, path[:, 2])

        # extend the time by the given number of points
        time_extended = np.arange(path.shape[0] + next_points)[:, np.newaxis]
        time_extended_transformed = self.t_poly.fit_transform(np.arange(time_extended.shape[0])[:, np.newaxis])

        # predict the next points
        pred_path = np.zeros(time_extended.shape[0] * 3).reshape(-1, 3)
        pred_path[:, 0] = xr.predict(np.arange(time_extended.shape[0])[:, np.newaxis])
        pred_path[:, 1] = yr.predict(time_extended_transformed)
        pred_path[:, 2] = zr.predict(np.arange(time_extended.shape[0])[:, np.newaxis])
        return pred_path
