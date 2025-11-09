from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from sqlmodel import select
from typing_extensions import TypeAlias

from cali._plate_viewer._graph_widgets import _SingleWellGraphWidget

if TYPE_CHECKING:
    from sqlmodel.sql import Select

from cali.sqlmodel._models import ROI

class AnalysisGroup(Enum):
    SINGLE_WELL = "single_well"
    MULTI_WELL = "multi_well"


AnalysisCallable: TypeAlias = Callable[
    [_SingleWellGraphWidget, dict, list[int] | None], Any
]

PRODUCTS: list[AnalysisProduct] = []


@dataclass
class AnalysisProduct:
    name: str
    group: AnalysisGroup
    analyzer: AnalysisCallable
    selector: Select = select(ROI)

    def __post_init__(self) -> None:
        if any(self.name == product.name for product in PRODUCTS):
            raise ValueError(f"AnalysisProduct with name '{self.name}' already exists.")
        PRODUCTS.append(self)


ray_traces = AnalysisProduct(
    name="Calcium Raw Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_plot_traces_data,
)
normalized_traces = AnalysisProduct(
    name="Calcium Normalized Traces",
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=partial(_plot_traces_data, normalize=True),
)

all_single_well_products = [x for x in PRODUCTS if x.group == AnalysisGroup.SINGLE_WELL]
