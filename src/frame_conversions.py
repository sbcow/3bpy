import numpy as np  # type: ignore
import sympy as sp


def convert_rotating_to_inertial_frame(vec, num_theta):
    """
    Conversion of a vector or vector of vectors by an angle which effectively rotates from a rotating to an inertial frame
    assuming a z-axis rotation.


    Parameters
    ----------
    vec: np.ndarray
        One or more vectors to be rotated
    num_theta: float
        The numerical values with which to rotate

    Returns
    -------
    np.ndarray
        The rotated vector or vector of vectors

    """
    theta = sp.symbols("theta")

    # Symbolic rotation matrix
    A = sp.Matrix(
        [
            [sp.cos(theta), -sp.sin(theta), 0],
            [sp.sin(theta), sp.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    # Check inverses == tranposes
    # sp.simplify(A.T * A)

    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, float)
    ):  # vec is 3x1 and theta is float
        return np.matmul(A.evalf(subs={theta: num_theta}), vec)
    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, np.ndarray)
    ):  # vec is 3x1 and theta is nx1 array
        return np.array([np.matmul(A.evalf(subs={theta: i}), vec) for i in num_theta])
    elif (
        isinstance(vec, np.ndarray)
        and vec.shape[0] > 1
        and vec.shape[1] == 3
        and isinstance(num_theta, float)
    ):  # vec is nx3 and theta is float
        return np.array(
            [np.matmul(A.evalf(subs={theta: num_theta}), vec_row) for vec_row in vec]
        )
    elif (
        isinstance(vec, np.ndarray)
        and vec.shape[0] > 1
        and vec.shape[1] == 3
        and isinstance(num_theta, np.ndarray)
    ):  # vec is nx3 and theta is nx1 array
        return np.array(
            [
                np.matmul(A.evalf(subs={theta: num_theta[it]}), vec_row)
                for it, vec_row in enumerate(vec)
            ]
        )
    else:
        raise RuntimeError(
            f"The provided vector: {vec} and theta: {num_theta} are malformed"
        )


def convert_inertial_to_rotating_frame(vec, num_theta):
    """
    Conversion of a vector or vector of vectors by an angle which effectively rotates from an
    inertial to a rotating frame assuming a z-axis rotation.


    Parameters
    ----------
    vec: np.ndarray
        One or more vectors to be rotated
    num_theta: float
        The numerical values with which to rotate

    Returns
    -------
    np.ndarray
        The rotated vector or vector of vectors

    """
    theta = sp.symbols("theta")

    # Symbolic rotation matrix
    A = sp.Matrix(
        [
            [sp.cos(theta), -sp.sin(theta), 0],
            [sp.sin(theta), sp.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    # Check inverses == tranposes
    # sp.simplify(A.T * A)

    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, float)
    ):  # vec is 3x1 and theta is float
        return np.matmul(A.T.evalf(subs={theta: num_theta}), vec)
    if (
        isinstance(vec, np.ndarray)
        and vec.shape == (3,)
        and isinstance(num_theta, np.ndarray)
    ):  # vec is 3x1 and theta is nx1 array
        return np.array([np.matmul(A.T.evalf(subs={theta: i}), vec) for i in num_theta])
    elif (
        isinstance(vec, np.ndarray) and vec.shape[0] > 1 and vec.shape[1] == 3
    ):  # vec is nx3 and theta is float
        return np.array(
            [np.matmul(A.T.evalf(subs={theta: num_theta}), vec_row) for vec_row in vec]
        )
    elif (
        isinstance(vec, np.ndarray)
        and vec.shape[0] > 1
        and vec.shape[1] == 3
        and isinstance(num_theta, np.ndarray)
    ):  # vec is nx3 and theta is nx1 array
        return np.array(
            [
                np.matmul(A.T.evalf(subs={theta: num_theta[it]}), vec_row)
                for it, vec_row in enumerate(vec)
            ]
        )
    else:
        raise RuntimeError(
            f"The provided vector: {vec} and theta: {num_theta} are malformed"
        )


def convert_rotating_inertial_frame_test():

    # Single case
    vec = np.array([1, 0, 0], dtype=float)
    theta = np.pi / 2
    exp_vec = np.array([0, 1, 0], dtype=float)
    obt_vec = convert_rotating_to_inertial_frame(vec, theta)
    orig_obt_vec = convert_inertial_to_rotating_frame(obt_vec, theta)
    assert np.isclose(exp_vec, obt_vec.astype(float)).all()
    assert np.isclose(vec, orig_obt_vec.astype(float)).all()

    # Vec case
    n_of_points = 20
    theta_points = np.linspace(0, 2 * np.pi, n_of_points)
    theta = np.pi / 2
    coordinates = np.zeros((n_of_points, 3))
    # coordinates = np.repeat(np.array([1, 0, 0]), n_of_points)
    for i in range(n_of_points):
        current_theta_point = theta_points[i]
        coordinates[i, :] = np.array(
            [np.cos(current_theta_point), np.sin(current_theta_point), 0]
        )

    exp_vec = coordinates
    rot_obt_vec = convert_rotating_to_inertial_frame(coordinates, theta)
    obt_vec = convert_inertial_to_rotating_frame(rot_obt_vec, theta)
    assert np.isclose(exp_vec, obt_vec.astype(float)).all()
