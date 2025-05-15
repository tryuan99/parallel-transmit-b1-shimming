"""The field file is the interface between a .fld file and a Pandas dataframe."""

from abc import ABC, abstractmethod

import pandas as pd


class FieldFile(ABC):
    """Field file.

    Attributes:
        data: Dataframe.
    """

    X_COLUMN = "x"
    Y_COLUMN = "y"
    Z_COLUMN = "z"

    def __init__(self, field_file: str) -> None:
        self.data = pd.read_csv(
            field_file,
            sep="\\s+",
            names=self.columns(),
            skiprows=2,
        )

    @abstractmethod
    def columns(self) -> list[str]:
        """Returns the list of columns."""


class HFieldFile(FieldFile):
    """H field file."""

    HX_REAL_COLUMN = "Re(Hx)"
    HX_IMAGINARY_COLUMN = "Im(Hx)"
    HY_REAL_COLUMN = "Re(Hy)"
    HY_IMAGINARY_COLUMN = "Im(Hy)"
    HZ_REAL_COLUMN = "Re(Hz)"
    HZ_IMAGINARY_COLUMN = "Im(Hz)"

    HX_COLUMN = "Hx"
    HY_COLUMN = "Hy"
    HZ_COLUMN = "Hz"

    def __init__(self, field_file: str) -> None:
        super().__init__(field_file)
        self.data[self.HX_COLUMN] = (self.data[self.HX_REAL_COLUMN] +
                                     1j * self.data[self.HX_IMAGINARY_COLUMN])
        self.data[self.HY_COLUMN] = (self.data[self.HY_REAL_COLUMN] +
                                     1j * self.data[self.HY_IMAGINARY_COLUMN])
        self.data[self.HZ_COLUMN] = (self.data[self.HZ_REAL_COLUMN] +
                                     1j * self.data[self.HZ_IMAGINARY_COLUMN])

    def columns(self) -> list[str]:
        """Returns the list of columns."""
        return [
            self.X_COLUMN,
            self.Y_COLUMN,
            self.Z_COLUMN,
            self.HX_REAL_COLUMN,
            self.HX_IMAGINARY_COLUMN,
            self.HY_REAL_COLUMN,
            self.HY_IMAGINARY_COLUMN,
            self.HZ_REAL_COLUMN,
            self.HZ_IMAGINARY_COLUMN,
        ]


class SarFieldFile(FieldFile):
    """SAR field file."""

    SAR_COLUMN = "SAR"

    def __init__(self, field_file: str) -> None:
        super().__init__(field_file)

    def columns(self) -> list[str]:
        """Returns the list of columns."""
        return [
            self.X_COLUMN,
            self.Y_COLUMN,
            self.Z_COLUMN,
            self.SAR_COLUMN,
        ]
