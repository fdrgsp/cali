from __future__ import annotations

from typing import cast

import useq
from pymmcore_widgets.useq_widgets import WellPlateWidget
from qtpy.QtWidgets import (
    QBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)


class PlatePlanWizard(QWizard):
    """A wizard for creating a plate plan."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Plate Plan Wizard")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        self.setMinimumWidth(600)
        self.setMinimumHeight(600)
        self.resize(600, 600)

        self._plate_plan: useq.WellPlatePlan | None = None

        # Set the button layout to include Cancel button
        self.setButtonLayout(
            [
                QWizard.WizardButton.Stretch,
                QWizard.WizardButton.CancelButton,
                QWizard.WizardButton.NextButton,
            ]
        )

        first_page = _QuestionPage(self)
        self._well_selection_page = _WellSelectionPage(self)
        self.addPage(first_page)
        self.addPage(self._well_selection_page)

        if cancel_button := self.button(QWizard.WizardButton.CancelButton):
            cancel_button.clicked.connect(self._on_cancel_clicked)

        # Connect the Finish button to close the wizard
        if finish_button := self.button(QWizard.WizardButton.FinishButton):
            finish_button.clicked.connect(self._on_finish_clicked)

    def value(self) -> useq.WellPlatePlan | None:
        """Return the plate plan if it was created, otherwise None."""
        return self._plate_plan

    def _on_cancel_clicked(self) -> None:
        """Handle the cancel button click - always close the wizard."""
        self._plate_plan = None
        self.close()

    def _on_finish_clicked(self) -> None:
        """Handle the finish button click."""
        plate_plan = self._well_selection_page.value()
        fovs = self._well_selection_page._fovs.value()
        self._plate_plan = plate_plan.replace(
            well_points_plan=useq.RandomPoints(num_points=fovs)
        )
        self.close()

    def dysplay_available_data_positions(self, n_positions: int | None) -> None:
        """Set the number of available data positions."""
        self._well_selection_page._pos_lbl.setText(
            f"(Available Positions: {n_positions})" if n_positions is not None else ""
        )


class _QuestionPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("Plate Plan Wizard")

        lbl = QLabel(
            "Did you use the `HCSWizard` to create a position list but manually "
            "modified it?\n\nIf you did, you can continue to the next step "
            "and select the wells you want to include in the plate plan."
        )
        layout = QVBoxLayout(self)
        layout.addWidget(lbl)

    def initializePage(self) -> None:
        """Initialize the page when it's shown."""
        super().initializePage()
        if wizard := self.wizard():
            wizard.setButtonText(QWizard.WizardButton.NextButton, "Yes")
            wizard.setButtonText(QWizard.WizardButton.CancelButton, "No")


class _WellSelectionPage(QWizardPage):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTitle("Well Selection")

        self._well_plate_widget = WellPlateWidget()

        fovs_wdg_layout = QHBoxLayout()
        fovs_wdg_layout.setContentsMargins(0, 0, 0, 0)
        fovs_lbl = QLabel("FOVs/Well:")
        fovs_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._fovs = QSpinBox(self)
        self._fovs.setMinimum(1)
        self._fovs.setMaximum(100)
        self._pos_lbl = QLabel()
        fovs_wdg_layout.addWidget(fovs_lbl)
        fovs_wdg_layout.addWidget(self._fovs, 1)
        fovs_wdg_layout.addWidget(self._pos_lbl)

        # Adjust the width of the plate label to match the FOVs label
        wpw_layout = cast("QBoxLayout", self._well_plate_widget.layout().itemAt(0))
        plate_lbl = cast("QLabel", wpw_layout.itemAt(0).widget())
        plate_lbl.setText("Well Plate:")

        wp_layout = cast("QVBoxLayout", self._well_plate_widget.layout())
        wp_layout.insertLayout(1, fovs_wdg_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(self._well_plate_widget)

    def initializePage(self) -> None:
        """Initialize the page when it's shown."""
        super().initializePage()
        if wizard := self.wizard():
            # Set the button layout for this page to only show Cancel and Finish
            wizard.setButtonLayout(
                [
                    QWizard.WizardButton.Stretch,
                    QWizard.WizardButton.CancelButton,
                    QWizard.WizardButton.FinishButton,
                ]
            )
            wizard.setButtonText(QWizard.WizardButton.FinishButton, "Finish")
            wizard.setButtonText(QWizard.WizardButton.CancelButton, "Cancel")

    def isComplete(self) -> bool:
        """Always return True so the Finish button is enabled."""
        return True

    def value(self) -> useq.WellPlatePlan:
        """Return the selected wells."""
        return self._well_plate_widget.value()
