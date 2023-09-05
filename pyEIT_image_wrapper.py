import math

import cv2
import numpy as np
import pyeit.mesh as mesh
from pyeit.mesh import PyEITMesh
from typing import Union


def pol2cart(r: Union[int, float], phi: Union[int, float]) -> tuple:
    """
    pol2cart converts cartesian coordinate system to polar coordinates.

    Parameters
    ----------
    r : Union[int, float]
        radius
    phi : Union[int, float]
        angle

    Returns
    -------
    tuple
        x and y position
    """
    x = r * np.cos(phi) + 100
    y = r * np.sin(phi) + 100
    return (int(x), int(y))


# Not the actual function to mesh an Image. This function is provided to test the following wrapper.
def geometry_to_img_wrot(
    objct: str = "circle", r: float = 0.5, phi: float = 0, d: float = 0.5
) -> np.ndarray:
    """
    geometry_to_img_wrot generates a ground truth image with rotation.

    Parameters
    ----------
    objct : str, optional
        select ["circle", "square", "triangle"], by default "circle"
    r : float, optional
        radius, by default 0.5
    phi : float, optional
        rotation angle, by default 0
    d : float, optional
        diameter, by default 0.5

    Returns
    -------
    np.ndarray
        meshed image
    """

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        return result

    def draw_circle(r, phi, d):
        IMG = np.zeros((200, 200))
        center_coordinates = pol2cart(r, phi * 0)  # *0 für die Rotation
        color = (1, 0, 0)
        thickness = -1
        IMG = cv2.circle(IMG, center_coordinates, int(d * 100), color, thickness)
        return IMG

    def draw_square(r, phi, d):
        IMG = np.zeros((200, 200))
        center_coordinates = pol2cart(r, phi * 0)  # *0 für die Rotation
        start_point = (
            int(center_coordinates[0] - int(d * 100)),
            center_coordinates[1] - int(d * 100),
        )
        end_point = (
            center_coordinates[0] + int(d * 100),
            center_coordinates[1] + int(d * 100),
        )
        color = (1, 0, 0)
        thickness = -1
        IMG = cv2.rectangle(IMG, start_point, end_point, color, thickness)
        return IMG

    def draw_triangle(r, phi, d):
        IMG = np.zeros((200, 200))
        center_coordinates = pol2cart(r, phi * 0)
        pt1 = (
            int(center_coordinates[0]),
            int(center_coordinates[1] - int((d) * 100)),
        )
        pt2 = (
            int(center_coordinates[0] + int((d) * 100)),
            int(center_coordinates[1]) + int((d) * 100),
        )
        pt3 = (
            int(center_coordinates[0] - int((d) * 100)),
            int(center_coordinates[1]) + int((d) * 100),
        )
        tri_edges = np.array([pt1, pt2, pt3])
        IMG = cv2.drawContours(IMG, [tri_edges], 0, (1, 0, 0), -1)
        return IMG

    r = int(r * 100)
    angle = phi  # Needed for rotation
    phi = math.radians(phi)  # Grad in Rad

    if objct == "circle":
        IMG = draw_circle(r, phi, d)
    if objct == "square":
        IMG = draw_square(r, phi, d)
    if objct == "triangle":
        IMG = draw_triangle(r, phi, d)

    IMG = rotate_image(IMG, angle)
    return IMG


def groundtruth_IMG_based(
    IMG: np.ndarray,
    n_el: int = 16,
    perm_empty_gnd: Union[int, float] = 1,
    perm_obj: Union[int, float] = 10,
    h0: float = 0.05,
) -> PyEITMesh:
    """
    groundtruth_IMG_based transforms a 200x200 picture to the pyEIT mesh.

    Parameters
    ----------
    IMG : np.ndarray
        image mask
    n_el : int, optional
        number of electrodes, by default 16
    perm_empty_gnd : Union[int, float], optional
        perm of the ground, by default 1
    perm_obj : Union[int, float], optional
        perm of the object, by default 10
    h0 : float, optional
        mesh refinement, by default 0.05

    Returns
    -------
    PyEITMesh
        mesh class of pyEIT
    """
    X_Y = np.array(np.where(IMG == 1))
    X = X_Y[1, :] - 100
    Y = (X_Y[0, :] - 100) * -1
    mesh_obj = mesh.create(n_el, h0=h0)
    pts = mesh_obj.element
    tri = mesh_obj.node
    perm = mesh_obj.perm
    tri_centers = np.mean(tri[pts], axis=1)
    mesh_x = np.round(tri_centers[:, 0] * 100)
    mesh_y = np.round(tri_centers[:, 1] * 100)
    Perm = np.ones(tri_centers.shape[0]) * perm_empty_gnd
    for i in range(len(X)):
        for j in range(len(mesh_x)):
            if X[i] == mesh_x[j] and Y[i] == mesh_y[j]:
                Perm[j] = perm_obj
    mesh_obj.perm = Perm
    return mesh_obj
