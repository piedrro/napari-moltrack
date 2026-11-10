"""Shared histogram range selection for localisation and track filters."""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QSignalBlocker


def clear_filter_plot(canvas):
    """Disconnect the previous selector before removing its plot."""
    selector = getattr(canvas, "filter_range", None)
    if selector is not None:
        selector.disconnect()
        canvas.filter_range = None
    canvas.clear()


def reset_filter_limits(values, minimum, maximum):
    """Reset both fields using their existing decimal precision."""
    if len(values):
        low, high = float(np.min(values)), float(np.max(values))
        # Enclose the full data range at the precision users can enter.
        # Nearest rounding can collapse small diffusion statistics to zero.
        low = float(
            Decimal(str(low)).quantize(
                Decimal(1).scaleb(-minimum.decimals()),
                rounding=ROUND_FLOOR,
            )
        )
        high = float(
            Decimal(str(high)).quantize(
                Decimal(1).scaleb(-maximum.decimals()),
                rounding=ROUND_CEILING,
            )
        )
    else:
        low, high = 0.0, 0.0
    with QSignalBlocker(minimum), QSignalBlocker(maximum):
        for field in (minimum, maximum):
            field.setRange(low, high)
        minimum.setValue(low)
        maximum.setValue(high)


class HistogramFilterRange:
    """Keep a movable value band and two spin boxes synchronised."""

    def __init__(self, canvas, plot, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        self.region = pg.LinearRegionItem(
            values=(minimum.value(), maximum.value()),
            bounds=(minimum.minimum(), maximum.maximum()),
            swapMode="block",
            brush=(255, 190, 0, 45),
            hoverBrush=(255, 190, 0, 75),
        )
        plot.addItem(self.region, ignoreBounds=True)
        self.region.sigRegionChanged.connect(self._from_region)
        self.region.sigRegionChangeFinished.connect(self._from_region)
        minimum.valueChanged.connect(self._from_minimum)
        maximum.valueChanged.connect(self._from_maximum)
        canvas.filter_range = self

    def _from_region(self):
        low, high = self.region.getRegion()
        with QSignalBlocker(self.minimum), QSignalBlocker(self.maximum):
            self.minimum.setValue(low)
            self.maximum.setValue(high)
        self._update_region()

    def _from_minimum(self):
        with QSignalBlocker(self.maximum):
            if self.minimum.value() > self.maximum.value():
                self.maximum.setValue(self.minimum.value())
        self._update_region()

    def _from_maximum(self):
        with QSignalBlocker(self.minimum):
            if self.maximum.value() < self.minimum.value():
                self.minimum.setValue(self.maximum.value())
        self._update_region()

    def _update_region(self):
        with QSignalBlocker(self.region):
            self.region.setRegion((self.minimum.value(), self.maximum.value()))

    def disconnect(self):
        self.region.sigRegionChanged.disconnect(self._from_region)
        self.region.sigRegionChangeFinished.disconnect(self._from_region)
        self.minimum.valueChanged.disconnect(self._from_minimum)
        self.maximum.valueChanged.disconnect(self._from_maximum)
