# Code taken from https://github.com/baurls/TileStitcher.
from __future__ import annotations


class Coordinate:
    def __init__(self, lat: float, lon: float):
        """
        Coordinate object.

        Parameters
        ----------
        lat : float
            latitude value (in range [-90°, 90°] )
        lon : float
            longitude value (in range [-180°, 180°] )
        """
        if not -90 <= lat <= 90:
            raise ValueError("Latitude must be in range [-90, 90]")
        if not -180 <= lon <= 180:
            raise ValueError("Longitude must be in range [-180, 180]")

        self.lat = lat
        self.lon = lon

    def __str__(self):
        return f"({self.lat}, {self.lon})"

    def __repr__(self):
        return str(self)


class GridIndex:
    def __init__(self, x: int, y: int, z: int):
        """GridIndex object.

        Parameters
        ----------
        x : int
        y : int
        z : int
            Zoom level
        """
        if not z >= 0:
            raise ValueError("Zoom level must be greater than or equal to 0")
        if not 0 <= x < 2**z:
            raise ValueError(f"X value must be in range [0, {2**z}]")
        if not 0 <= y < 2**z:
            raise ValueError(f"Y value must be in range [0, {2**z}]")
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.z}, {self.x}, {self.y})"

    def __repr__(self):
        return str(self)


class GridBoundingBox:
    def __init__(self, cell1: GridIndex, cell2: GridIndex):
        """
        GridBoundingBox object.

        Parameters
        ----------
        cell1 : GridIndex
        cell2 : GridIndex
        """
        if cell1.z != cell2.z:
            raise NotImplementedError("Can't calculate a grid on different scales yet")

        start_x = min(cell1.x, cell2.x)
        end_x = max(cell1.x, cell2.x)
        start_y = min(cell1.y, cell2.y)
        end_y = max(cell1.y, cell2.y)
        self.z = cell1.z
        self.lower_corner = GridIndex(start_x, start_y, z=self.z)
        self.upper_corner = GridIndex(end_x, end_y, z=self.z)

    @property
    def covered_cells(self):
        return (self.upper_corner.x - self.lower_corner.x + 1) * (
            self.upper_corner.y - self.lower_corner.y + 1
        )

    @property
    def x_range(self):
        return range(self.lower_corner.x, self.upper_corner.x + 1)

    @property
    def y_range(self):
        return range(self.lower_corner.y, self.upper_corner.y + 1)

    def __str__(self):
        return f"[{self.lower_corner}x{self.upper_corner}]"

    def __repr__(self):
        return str(self)
