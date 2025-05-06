"""The B1 field data manages and analyzes the simulated B field."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

from simulation.field_file import FieldFile, HFieldFile, SarFieldFile
from utils import constants


class B1FieldData:
    """B1 field data.
    
    Attributes:
        data: B1 field within the region of interest.
        full_data: B1 field within the entire bounding box.
    """

    VALID_COLUMN = "Valid"

    BX_COLUMN = "Bx"
    BY_COLUMN = "By"
    BZ_COLUMN = "Bz"

    BX_MAG_COLUMN = "Mag(Bx)"
    BY_MAG_COLUMN = "Mag(By)"
    BZ_MAG_COLUMN = "Mag(Bz)"

    B1_COLUMN = "B1"
    B1_MAG_COLUMN = "Mag(B1)"

    def __init__(self, field: HFieldFile, mask: SarFieldFile) -> None:
        self.full_data = pd.merge(
            field.data,
            mask.data,
            on=[
                FieldFile.X_COLUMN,
                FieldFile.Y_COLUMN,
                FieldFile.Z_COLUMN,
            ],
        )
        self.full_data[self.VALID_COLUMN] = mask.data[mask.SAR_COLUMN] != 0
        self.full_data = self._convert_to_b_field(self.full_data)
        self.data = self.full_data[self.full_data[self.VALID_COLUMN]]

    def plot_magnitude(self) -> None:
        """Plots the B1 field magnitude."""
        x_min = self.full_data[FieldFile.X_COLUMN].min()
        x_max = self.full_data[FieldFile.X_COLUMN].max()
        y_min = self.full_data[FieldFile.Y_COLUMN].min()
        y_max = self.full_data[FieldFile.Y_COLUMN].max()
        num_x_values = len(self.full_data[FieldFile.X_COLUMN].unique())
        num_y_values = len(self.full_data[FieldFile.Y_COLUMN].unique())
        delta_x = (x_max - x_min) / (num_x_values - 1)
        delta_y = (y_max - y_min) / (num_y_values - 1)

        b1_magnitude = self.full_data[self.B1_MAG_COLUMN].to_numpy()
        b1_magnitude[~self.full_data[self.VALID_COLUMN]] = np.nan
        b1_magnitude = b1_magnitude.reshape(num_x_values, num_y_values)

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

    def plot_normalized_magnitude(self) -> None:
        """Plots the normalized B1 field magnitude.
        
        The B1 field is normalized by the mean of the B1 magnitude.
        """
        x_min = self.full_data[FieldFile.X_COLUMN].min()
        x_max = self.full_data[FieldFile.X_COLUMN].max()
        y_min = self.full_data[FieldFile.Y_COLUMN].min()
        y_max = self.full_data[FieldFile.Y_COLUMN].max()
        num_x_values = len(self.full_data[FieldFile.X_COLUMN].unique())
        num_y_values = len(self.full_data[FieldFile.Y_COLUMN].unique())
        delta_x = (x_max - x_min) / (num_x_values - 1)
        delta_y = (y_max - y_min) / (num_y_values - 1)

        b1_magnitude = self.full_data[self.B1_MAG_COLUMN].to_numpy()
        b1_magnitude[~self.full_data[self.VALID_COLUMN]] = np.nan
        b1_magnitude = b1_magnitude.reshape(num_x_values, num_y_values)
        mean_b1_magnitude = self.full_data[self.B1_MAG_COLUMN].mean()
        normalized_b1_magnitude = ((b1_magnitude - mean_b1_magnitude) /
                                   mean_b1_magnitude)

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

    def calculate_inhomogeneity(self) -> float:
        """Calculates the B1 inhomogeneity.
        
        The B1 inhomogeneity is defined as the standard deviation of the B1
        magnitude divided by the mean of the B1 magnitude.

        Returns:
            The B1 inhomogeneity.
        """
        return (self.data[self.B1_MAG_COLUMN].std() /
                self.data[self.B1_MAG_COLUMN].mean())

    @staticmethod
    def _convert_to_b_field(field: pd.DataFrame) -> pd.DataFrame:
        """Converts the H field to a B field.
        
        Returns:
            The converted dataframe.
        """
        field[B1FieldData.BX_COLUMN] = (field[HFieldFile.HX_COLUMN] *
                                        constants.VACUUM_PERMEABILITY)
        field[B1FieldData.BY_COLUMN] = (field[HFieldFile.HY_COLUMN] *
                                        constants.VACUUM_PERMEABILITY)
        field[B1FieldData.BZ_COLUMN] = (field[HFieldFile.HZ_COLUMN] *
                                        constants.VACUUM_PERMEABILITY)

        field[B1FieldData.BX_MAG_COLUMN] = np.abs(field[B1FieldData.BX_COLUMN])
        field[B1FieldData.BY_MAG_COLUMN] = np.abs(field[B1FieldData.BY_COLUMN])
        field[B1FieldData.BZ_MAG_COLUMN] = np.abs(field[B1FieldData.BZ_COLUMN])

        field[B1FieldData.B1_COLUMN] = (field[B1FieldData.BX_MAG_COLUMN] +
                                        1j * field[B1FieldData.BY_MAG_COLUMN])
        field[B1FieldData.B1_MAG_COLUMN] = np.abs(field[B1FieldData.B1_COLUMN])
        return field
