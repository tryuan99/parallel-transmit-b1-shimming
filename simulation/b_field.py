"""The B field manages and analyzes the simulated B field within a region of
interest.
"""
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from simulation.field_file import FieldFile, HFieldFile, SarFieldFile
from utils import constants


class BField:
    """B field.
    
    Attributes:
        coordinates: The x, y, and z coordinates.
        data: Complex B field at each coordinate.
    """

    def __init__(self,
                 field: HFieldFile = None,
                 mask: SarFieldFile = None,
                 coordinates: np.ndarray = None,
                 data: np.ndarray = None) -> None:
        if coordinates is not None and data is not None:
            self.coordinates = coordinates
            self.data = data
        else:
            data = pd.merge(
                field.data,
                mask.data,
                on=[
                    FieldFile.X_COLUMN,
                    FieldFile.Y_COLUMN,
                    FieldFile.Z_COLUMN,
                ],
            )
            # Only keep the data within the region of interest.
            valid_data = data[mask.data[mask.SAR_COLUMN] != 0]
            self.coordinates = valid_data[[
                HFieldFile.X_COLUMN,
                HFieldFile.Y_COLUMN,
                HFieldFile.Z_COLUMN,
            ]].to_numpy()
            # Convert from the magnetic field strength H to the magnetic flux
            # density B.
            self.data = valid_data[[
                HFieldFile.HX_COLUMN,
                HFieldFile.HY_COLUMN,
                HFieldFile.HZ_COLUMN,
            ]].to_numpy() * constants.VACUUM_PERMEABILITY

    def x(self) -> np.ndarray:
        """Returns the x-coordinates."""
        return self.coordinates[:, 0]

    def y(self) -> np.ndarray:
        """Returns the y-coordinates."""
        return self.coordinates[:, 1]

    def z(self) -> np.ndarray:
        """Returns the z-coordinates."""
        return self.coordinates[:, 2]

    def bx(self) -> np.ndarray:
        """Returns the x-component of the B field."""
        return self.data[:, 0]

    def by(self) -> np.ndarray:
        """Returns the y-component of the B field."""
        return self.data[:, 1]

    def bz(self) -> np.ndarray:
        """Returns the z-component of the B field."""
        return self.data[:, 2]

    def bx_magnitude(self) -> np.ndarray:
        """Returns the magnitude of the x-component of the B field."""
        return np.abs(self.bx())

    def by_magnitude(self) -> np.ndarray:
        """Returns the magnitude of the y-component of the B field."""
        return np.abs(self.by())

    def bz_magnitude(self) -> np.ndarray:
        """Returns the magnitude of the z-component of the B field."""
        return np.abs(self.bz())

    def b1(self) -> np.ndarray:
        """Returns the complex B1 field."""
        return self.bx_magnitude() + 1j * self.by_magnitude()

    def b1_magnitude(self) -> np.ndarray:
        """Returns the magnitude of the B1 field."""
        return np.abs(self.b1())

    def b1_normalized_magnitude(self) -> np.ndarray:
        """Returns the normalized magnitude of the B1 field.
        
        The B1 field magnitude is centered around zero and normaliezd by the
        mean of the B1 field magnitude.
        """
        b1_magnitude = self.b1_magnitude()
        mean_b1_magnitude = np.mean(b1_magnitude)
        return (b1_magnitude - mean_b1_magnitude) / mean_b1_magnitude

    def b1_inhomogeneity(self) -> float:
        """Calculates the B1 inhomogeneity.
        
        The B1 inhomogeneity is defined as the standard deviation of the B1
        magnitude divided by the mean of the B1 magnitude.

        Returns:
            The B1 inhomogeneity.
        """
        b1_magnitude = self.b1_magnitude()
        return np.std(b1_magnitude) / np.mean(b1_magnitude)

    @classmethod
    def sum(cls, fields: list[Self], weights: np.ndarray = None) -> Self:
        """Returns a B field instance representing the complex weighted sum of
        the given list of B fields.
        
        Args:
            fields: List of B fields.
            weights: Weights for each B field.
        
        Returns:
            The complex weighted sum of the B fields.
        """
        if weights is None:
            weights = np.ones(len(fields))
        coordinates = fields[0].coordinates
        return cls(
            coordinates=coordinates,
            data=np.sum(
                [weight * field.data for weight, field in zip(weights, fields)],
                axis=0,
            ),
        )

    def plot_b1_magnitude(self) -> None:
        """Plots the B1 field magnitude."""
        x_min = np.min(self.x())
        x_max = np.max(self.x())
        y_min = np.min(self.y())
        y_max = np.max(self.y())
        num_x_values = len(np.unique(self.x()))
        num_y_values = len(np.unique(self.y()))
        delta_x = (x_max - x_min) / (num_x_values - 1)
        delta_y = (y_max - y_min) / (num_y_values - 1)

        b1_magnitude = np.full((num_x_values, num_y_values), np.nan)
        x_indices = np.round((self.x() - x_min) / delta_x).astype(np.int64)
        y_indices = np.round((self.y() - y_min) / delta_y).astype(np.int64)
        b1_magnitude[x_indices, y_indices] = self.b1_magnitude()

        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(figsize=(12, 6))
        image = ax.imshow(
            b1_magnitude.T,
            cmap="jet",
            origin="lower",
            extent=(
                x_min - delta_x,
                x_max - delta_x,
                y_min - delta_y,
                y_max - delta_y,
            ),
        )
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        plt.colorbar(image, label="B1 field magnitude")
        plt.show()

    def plot_normalized_b1_magnitude(self) -> None:
        """Plots the normalized B1 field magnitude."""
        x_min = np.min(self.x())
        x_max = np.max(self.x())
        y_min = np.min(self.y())
        y_max = np.max(self.y())
        num_x_values = len(np.unique(self.x()))
        num_y_values = len(np.unique(self.y()))
        delta_x = (x_max - x_min) / (num_x_values - 1)
        delta_y = (y_max - y_min) / (num_y_values - 1)

        normalized_b1_magnitude = np.full((num_x_values, num_y_values), np.nan)
        x_indices = ((self.x() - x_min) / delta_x).astype(np.int64)
        y_indices = ((self.y() - y_min) / delta_y).astype(np.int64)
        normalized_b1_magnitude[x_indices,
                                y_indices] = (self.b1_normalized_magnitude())

        plt.style.use(["science", "grid"])
        fig, ax = plt.subplots(figsize=(12, 6))
        image = ax.imshow(
            normalized_b1_magnitude.T,
            cmap="jet",
            vmin=-1,
            vmax=1,
            origin="lower",
            extent=(
                x_min - delta_x,
                x_max - delta_x,
                y_min - delta_y,
                y_max - delta_y,
            ),
        )
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        plt.colorbar(image, label="Normalized B1 field magnitude")
        plt.show()
