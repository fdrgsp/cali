"""Auditable links between analysis results and the extraction they used."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import UniqueConstraint
from sqlmodel import JSON, Column, Field, SQLModel, select

if TYPE_CHECKING:
    from sqlmodel import Session

    from ._model import CaliResult


class MigrationIssue(SQLModel, table=True):  # type: ignore[call-arg, unused-ignore]
    """An unresolved historical source stays readable and explicitly auditable."""

    __tablename__ = "migration_issue"
    __table_args__ = (UniqueConstraint("analysis_result_id", "code"),)

    id: int | None = Field(default=None, primary_key=True)
    analysis_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", ondelete="CASCADE", index=True
    )
    code: str
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    resolved: bool = False


def record_result_sources(
    session: "Session", result: "CaliResult", source_ids: set[int]
) -> None:
    """Record a common source, retaining an audit when a legacy run spans sources."""
    if not source_ids:
        return
    previous = result.source_extraction_result_id
    if previous is not None:
        source_ids = source_ids | {previous}
    if len(source_ids) == 1 and result.legacy_trace_resolution != "multiple_sources":
        source_id = next(iter(source_ids))
        result.source_extraction_result_id = source_id
        result.legacy_trace_resolution = (
            "self" if source_id == result.id else "source_selected"
        )
    else:
        result.source_extraction_result_id = None
        result.legacy_trace_resolution = "multiple_sources"
        issue = session.exec(
            select(MigrationIssue).where(
                MigrationIssue.analysis_result_id == result.id,
                MigrationIssue.code == "multiple_extraction_sources",
            )
        ).first()
        if issue is None:
            issue = MigrationIssue(
                analysis_result_id=result.id, code="multiple_extraction_sources"
            )
        prior = issue.details.get("source_result_ids", [])
        issue.details = {"source_result_ids": sorted(source_ids | set(prior))}
        session.add(issue)
    session.add(result)
