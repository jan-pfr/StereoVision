import numpy as np


class Triangulation:
    """
    The method of this class uses the centers of the contours of the object from the Left and Right images
    to calculate the distance from the object to the baseline on which the cameras are located.
    For the calculation, the images must have the same size.
    """

    def __init__(self):
        pass

    def find_depth(self, right_point, left_point, frame_right, frame_left, baseline, alpha) -> float:
        """
        Caculate the depth from the given coordinates with the disparity.

        :param right_point: right frame center point from object
        :param left_point: left frame center point from object
        :param frame_right: right frame as np.array
        :param frame_left: left frame as np.array
        :param baseline: distance between the cameras
        :param alpha: field of view
        :return: distance in cm
        """

        # receive the height and width of the images
        height_right, width_right, depth_right = frame_right.shape
        height_left, width_left, depth_left = frame_left.shape

        if width_right == width_left:
            # focal length in pixel based on the field of view
            # Source: https://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/
            f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

        else:
            print('Left and right camera frames do not have the same pixel width')

        x_right = right_point[0]
        x_left = left_point[0]

        # calculate disparity
        disparity = x_left - x_right  # in pixel

        # calculate depth
        depth = (baseline * f_pixel) / disparity  # in cm

        return abs(depth)
