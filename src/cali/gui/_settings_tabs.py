"""Consistent settings pages with visible guidance and keyboard-accessible checks."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QScrollArea,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def guidance(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setContentsMargins(0, 0, 0, 6)
    return label


def settings_section(title: str, description: str, *widgets: QWidget) -> QGroupBox:
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(12, 14, 12, 12)
    layout.setSpacing(8)
    if description:
        layout.addWidget(guidance(description, group))
    for widget in widgets:
        layout.addWidget(widget)
    return group


class _SettingsTabs(QTabWidget):
    """Viewing a page never changes whether that computation is selected."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDocumentMode(True)
        self.setUsesScrollButtons(True)

    def add_page(
        self,
        title: str,
        description: str,
        *widgets: QWidget,
        checkbox: QCheckBox | None = None,
    ) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        if description:
            layout.addWidget(guidance(description, content))
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        page = QScrollArea(self)
        page.setWidgetResizable(True)
        page.setFrameShape(QScrollArea.Shape.NoFrame)
        page.setWidget(content)
        index = self.addTab(page, title)
        if checkbox is not None:
            checkbox.setText("")
            checkbox.setAccessibleName(f"Include {title}")
            checkbox.setToolTip(checkbox.toolTip() or f"Include {title} in this run.")
            self.tabBar().setTabButton(index, QTabBar.ButtonPosition.LeftSide, checkbox)
            checkbox.clicked.connect(lambda: self.setCurrentWidget(page))
        return page
