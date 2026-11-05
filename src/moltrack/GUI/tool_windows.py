from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .export_ui import Ui_Export
from .heatmaps_ui import Ui_Heatmaps
from .import_project_ui import Ui_ImportProject
from .localisation_ui import Ui_Localisation
from .segmentation_ui import Ui_Segmentation
from .tracking_ui import Ui_Tracking
from .utilities_ui import Ui_Utilities

TOOL_SECTIONS = (
    ("Import & Project", Ui_ImportProject, "tabWidget_4"),
    ("Segmentation", Ui_Segmentation, "tabWidget_7"),
    ("Localisation", Ui_Localisation, "tabWidget_6"),
    ("Tracking", Ui_Tracking, "tabWidget_5"),
    ("Heatmaps", Ui_Heatmaps, "tabWidget_3"),
    ("Export", Ui_Export, "tabWidget_2"),
    ("Utilities", Ui_Utilities, "tabWidget_8"),
)


class ToolWindow(QWidget):
    """One section's form in a window owned by the MolTrack launcher."""

    def __init__(self, title, form_class, pages_name, parent):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(f"MolTrack — {title}")

        content = QWidget()
        self.form = form_class()
        self.form.setupUi(content)
        self.pages = getattr(self.form, pages_name)
        self.pages.tabBar().hide()

        self.tool_selector = QComboBox()
        for index in range(self.pages.count()):
            self.tool_selector.addItem(self.pages.tabText(index))
        self.tool_selector.setCurrentIndex(self.pages.currentIndex())
        self.tool_selector.currentIndexChanged.connect(
            self.pages.setCurrentIndex
        )
        self.pages.currentChanged.connect(self.tool_selector.setCurrentIndex)

        label = QLabel("&Tool")
        label.setBuddy(self.tool_selector)
        navigation = QFormLayout()
        navigation.addRow(label, self.tool_selector)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addLayout(navigation)
        layout.addWidget(scroll)
        self.resize(475, 625)

    def open_or_focus(self):
        if self.isMinimized():
            self.setWindowState(
                self.windowState() & ~Qt.WindowState.WindowMinimized
            )
        self.show()
        self.raise_()
        self.activateWindow()
