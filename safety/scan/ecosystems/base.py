from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from safety_schemas.models import (
    Ecosystem,
    FileType,
    ConfigModel,
    DependencyResultModel,
)
from typer import FileTextWrite

NOT_IMPLEMENTED = "Not implemented funtion"


class SkipReason(str, Enum):
    """
    Reasons why a requirement line was kept out of the scan set.

    The scan set mirrors what ``pip install`` would resolve in the current
    environment: lines pip would not install here, and lines Safety cannot map
    to a released package version, are skipped with a reason so the report
    stays traceable instead of silently dropping them.
    """

    ENVIRONMENT_MARKER = "environment-marker"
    VCS_DEPENDENCY = "vcs-dependency"
    EDITABLE_DEPENDENCY = "editable-dependency"
    LOCAL_PATH_DEPENDENCY = "local-path-dependency"
    DIRECT_URL_DEPENDENCY = "direct-url-dependency"
    UNRESOLVABLE_DIRECT_REFERENCE = "unresolvable-direct-reference"
    UNPARSEABLE_LINE = "unparseable-line"
    UNSUPPORTED_FORMAT = "unsupported-dependency-format"
    UNRESOLVED_INCLUDE = "unresolved-include"
    CIRCULAR_INCLUDE = "circular-include"
    FILE_PARSE_ERROR = "file-parse-error"


@dataclass
class SkippedRequirement:
    """
    A requirement line that was not scanned, along with the reason why.

    Attributes:
        raw (str): The original requirement line, as written in the file.
        reason (SkipReason): The category of the skip.
        detail (str): Human-readable explanation for the skip.
        source (Optional[str]): The file the line was read from (differs from
            the scanned file when it comes from an included ``-r`` file).
        line_number (Optional[int]): The 1-based line number in ``source``.
    """

    raw: str
    reason: SkipReason
    detail: str
    source: Optional[str] = None
    line_number: Optional[int] = None

    def to_dict(self) -> dict:
        """
        Serializes the skipped requirement for JSON reports.

        Returns:
            dict: A JSON-serializable representation.
        """
        return {
            "raw": self.raw,
            "reason": self.reason.value,
            "detail": self.detail,
            "source": self.source,
            "line_number": self.line_number,
        }


class Inspectable(ABC):
    """
    Abstract base class defining the interface for objects that can be inspected for dependencies.
    """

    @abstractmethod
    def inspect(self, config: ConfigModel) -> None:
        """
        Inspects the object and populates the dependency analysis results in-place.

        Args:
            config (ConfigModel): The configuration model for inspection.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)


class Remediable(ABC):
    """
    Abstract base class defining the interface for objects that can be remediated.
    """

    @abstractmethod
    def remediate(self) -> None:
        """
        Remediates the object to address any detected issues.
        """
        raise NotImplementedError(NOT_IMPLEMENTED)


class InspectableFile(Inspectable):
    """
    Represents an inspectable file within a specific ecosystem and file type.
    """

    def __init__(self, file: FileTextWrite):
        """
        Initializes an InspectableFile instance.

        Args:
            file (FileTextWrite): The file to be inspected.
        """
        self.file = file
        self.ecosystem: Ecosystem
        self.file_type: FileType
        self.dependency_results: DependencyResultModel = DependencyResultModel(
            dependencies=[]
        )
        self.skipped_dependencies: List[SkippedRequirement] = []
