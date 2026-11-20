from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import logging
import sys
from typing import Dict, Generator, List, Optional, Tuple

from safety_schemas.models import FileType, PythonDependency
from safety_schemas.models.package import PythonSpecification
from ..base import InspectableFile, SkippedRequirement, SkipReason
from .parsing import (
    collect_pipfile_lines,
    collect_pyproject_lines,
    collect_requirement_lines,
    collect_setup_cfg_lines,
    normalize_pipfile_lock,
)
from dparse import parse, filetypes

from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse as parse_version
from packaging.utils import canonicalize_name

LOG = logging.getLogger(__name__)


def get_closest_ver(
    versions: List[str], version: Optional[str], spec: SpecifierSet
) -> dict:
    """
    Gets the closest version to the specified version within a list of versions.

    Args:
        versions (List[str]): The list of versions.
        version (Optional[str]): The target version.
        spec (SpecifierSet): The version specifier set.

    Returns:
        dict: A dictionary containing the upper and lower closest versions.
    """
    results: dict[str, str | Version | None] = {"upper": None, "lower": None}

    if (not version and not spec) or not versions:
        return results

    sorted_versions = sorted(versions, key=lambda ver: parse_version(ver), reverse=True)

    if not version:
        sorted_versions = spec.filter(sorted_versions, prereleases=False)

        upper = None
        lower = None

        try:
            sorted_versions = list(sorted_versions)
            upper = sorted_versions[0]
            lower = sorted_versions[-1]
            results["upper"] = upper
            results["lower"] = lower if upper != lower else None
        except IndexError:
            pass

        return results

    current_v = parse_version(version)

    for v in sorted_versions:
        index = parse_version(v)

        if index > current_v:
            results["upper"] = index

        if index < current_v:
            results["lower"] = index
            break

    return results


def is_pinned_requirement(spec: SpecifierSet) -> bool:
    """
    Checks if a requirement is pinned.

    Args:
        spec (SpecifierSet): The version specifier set.

    Returns:
        bool: True if the requirement is pinned, False otherwise.
    """
    if not spec or len(spec) != 1:
        return False

    specifier = next(iter(spec))

    return (
        specifier.operator == "==" and "*" != specifier.version[-1]
    ) or specifier.operator == "==="


def find_version(requirements: List[PythonSpecification]) -> Optional[str]:
    """
    Finds the version of a requirement.

    Args:
        requirements (List[PythonSpecification]): The list of requirements.

    Returns:
        Optional[str]: The version if found, otherwise None.
    """
    ver = None

    if len(requirements) != 1:
        return ver

    specs = requirements[0].specifier

    if is_pinned_requirement(specs):
        ver = next(iter(requirements[0].specifier)).version

    return ver


def is_supported_by_parser(path: str) -> bool:
    """
    Checks if the file path is supported by the parser.

    Args:
        path (str): The file path.

    Returns:
        bool: True if supported, False otherwise.
    """
    supported_types = (
        ".txt",
        ".in",
        ".yml",
        ".ini",
        "Pipfile",
        "Pipfile.lock",
        "setup.cfg",
        "poetry.lock",
    )
    return path.endswith(supported_types)


def parse_requirement(dep: str, found: Optional[str]) -> PythonSpecification:
    """
    Parses a requirement and creates a PythonSpecification object.

    Args:
        dep (str): The dependency string.
        found (Optional[str]): The found path.

    Returns:
        PythonSpecification: The parsed requirement.
    """
    req = PythonSpecification(dep)
    req.found = Path(found).resolve() if found else None

    if req.specifier == SpecifierSet(""):
        req.specifier = SpecifierSet(">=0")

    return req


def read_requirements(
    fh, resolve: bool = True
) -> Generator[PythonDependency, None, None]:
    """
    Reads requirements from a file-like object and (optionally) from referenced files.

    Args:
        fh: The file-like object to read from.
        resolve (bool): Whether to resolve referenced files.

    Returns:
        Generator[PythonDependency, None, None]: A generator of PythonDependency objects.
    """
    is_temp_file = not hasattr(fh, "name")
    path = None

    if not is_temp_file and is_supported_by_parser(fh.name):
        path = fh.name

    content = fh.read()
    dependencies, _ = parse_dependency_content(content, path, resolve)

    yield from dependencies


def read_dependencies(
    fh, resolve: bool = True
) -> Generator[PythonDependency, None, None]:
    """
    Reads dependencies from a file-like object.

    Args:
        fh: The file-like object to read from.
        resolve (bool): Whether to resolve referenced files.

    Returns:
        Generator[PythonDependency, None, None]: A generator of PythonDependency objects.
    """
    dependencies, _ = read_dependencies_with_skipped(fh, resolve)

    yield from dependencies


def read_dependencies_with_skipped(
    fh, resolve: bool = True
) -> Tuple[List[PythonDependency], List[SkippedRequirement]]:
    """
    Reads dependencies from a file-like object, also returning the lines
    that were skipped with their reasons.

    Args:
        fh: The file-like object to read from.
        resolve (bool): Whether to resolve referenced files.

    Returns:
        Tuple[List[PythonDependency], List[SkippedRequirement]]: The
        dependencies to scan and the skipped entries.
    """
    content = fh.read()

    return parse_dependency_content(content, fh.name, resolve)


def _run_dparse(
    content: str,
    path: Optional[str],
    file_type: Optional[str],
    resolve: bool,
    skipped: List[SkippedRequirement],
) -> list:
    """
    Runs dparse on the given content, converting parser crashes into a
    traceable skipped entry instead of aborting the whole scan.

    Args:
        content (str): The content to parse.
        path (Optional[str]): The file path, used for parser detection.
        file_type (Optional[str]): An explicit dparse file type.
        resolve (bool): Whether dparse should resolve referenced files.
        skipped (List[SkippedRequirement]): Accumulator for skipped entries.

    Returns:
        list: The parsed dparse dependencies, or an empty list on failure.
    """
    try:
        dependency_file = parse(
            content, path=path, resolve=resolve, file_type=file_type
        )
    except Exception as exc:
        LOG.warning("Unable to parse %s: %s", path or "provided input", exc)
        skipped.append(
            SkippedRequirement(
                raw=str(path or "provided input"),
                reason=SkipReason.FILE_PARSE_ERROR,
                detail=f"The file could not be parsed: {exc}",
                source=path,
            )
        )
        return []

    return dependency_file.resolved_dependencies


def _marker_applies(specification: PythonSpecification) -> bool:
    """
    Evaluates a specification's environment marker against the current
    environment, mirroring what pip would install locally.

    Args:
        specification (PythonSpecification): The specification to check.

    Returns:
        bool: True if the requirement applies to this environment. Markers
        that cannot be evaluated are kept (conservative).
    """
    marker = specification.marker

    if marker is None:
        return True

    try:
        return bool(marker.evaluate())
    except Exception:
        LOG.warning(
            "Unable to evaluate environment marker '%s' of '%s'; keeping it.",
            marker,
            specification.raw,
        )
        return True


def _key_of(specification: PythonSpecification) -> tuple:
    """
    Builds the deduplication key of a specification (same specifier, marker
    and direct URL mean the same effective requirement).

    Args:
        specification (PythonSpecification): The specification.

    Returns:
        tuple: The key used in ``specifications_by_key``.
    """
    return (
        str(specification.specifier),
        str(specification.marker) if specification.marker else None,
        specification.url,
    )


def _can_merge(first: PythonSpecification, second: PythonSpecification) -> bool:
    """
    Checks whether two specifications of the same package can be folded into
    one, i.e. neither has a direct URL and both are guarded by the same
    environment marker.

    Args:
        first (PythonSpecification): One specification.
        second (PythonSpecification): The other specification.

    Returns:
        bool: True if they describe the same installable requirement.
    """
    if first.url or second.url:
        return False

    first_marker = str(first.marker) if first.marker else None
    second_marker = str(second.marker) if second.marker else None

    return first_marker == second_marker


def _pin_version(specification: PythonSpecification) -> Optional[str]:
    """
    Returns the single version a pinned specification installs.

    Args:
        specification (PythonSpecification): The specification.

    Returns:
        Optional[str]: The pinned version, or None if the specification is not
        pinned.
    """
    if not specification.is_pinned():
        return None

    return next(iter(specification.specifier)).version


def _matching_pin(
    specifications_by_key: Dict[tuple, PythonSpecification],
    specification: PythonSpecification,
) -> Optional[PythonSpecification]:
    """
    Finds an already-kept pin that the given specification narrows.

    Args:
        specifications_by_key (Dict[tuple, PythonSpecification]): The
            specifications kept so far for this package.
        specification (PythonSpecification): The new specification.

    Returns:
        Optional[PythonSpecification]: The pin that satisfies the new
        requirement, if any.
    """
    for kept in specifications_by_key.values():
        version = _pin_version(kept)

        if version is None or not _can_merge(kept, specification):
            continue

        try:
            if specification.specifier.contains(version, prereleases=True):
                return kept
        except Exception:
            continue

    return None


def _matching_ranges(
    specifications_by_key: Dict[tuple, PythonSpecification],
    specification: PythonSpecification,
) -> List[PythonSpecification]:
    """
    Finds the already-kept ranges that the new pin satisfies.

    Args:
        specifications_by_key (Dict[tuple, PythonSpecification]): The
            specifications kept so far for this package.
        specification (PythonSpecification): The new (pinned) specification.

    Returns:
        List[PythonSpecification]: The ranges the pin satisfies (all of them
        are redundant once the pin is kept).
    """
    version = _pin_version(specification)

    if version is None:
        return []

    matches = []

    for kept in specifications_by_key.values():
        if _pin_version(kept) is not None or not _can_merge(kept, specification):
            continue

        try:
            if kept.specifier.contains(version, prereleases=True):
                matches.append(kept)
        except Exception:
            continue

    return matches


def _merge_extras(target: PythonSpecification, other: PythonSpecification) -> None:
    """
    Merges the extras of ``other`` into ``target``.

    Args:
        target (PythonSpecification): The specification that is kept.
        other (PythonSpecification): The specification being folded in.
    """
    try:
        target.extras = target.extras | other.extras
    except Exception:
        pass


def _build_dependencies(
    dparse_deps: list,
    found: Path,
    absolute_path: Optional[Path],
    raw_overrides: Dict[str, str],
    skipped: List[SkippedRequirement],
    line_locations: Optional[Dict[str, List[Tuple[str, int]]]] = None,
) -> Generator[PythonDependency, None, None]:
    """
    Builds the scan-ready dependencies from parsed requirements.

    Specifications whose environment marker does not match the current
    environment are skipped (pip would not install them), and duplicate
    specifications of the same package are collapsed so vulnerabilities are
    not counted multiple times.

    Args:
        dparse_deps (list): The parsed dparse dependencies.
        found (Path): The file the dependencies were found in.
        absolute_path (Optional[Path]): The resolved file path.
        raw_overrides (Dict[str, str]): Maps rewritten lines to their
            original raw line.
        skipped (List[SkippedRequirement]): Accumulator for skipped entries.
        line_locations (Optional[Dict[str, List[Tuple[str, int]]]]): Maps a
            requirement line to the (source, line number) it came from.

    Yields:
        PythonDependency: The deduplicated dependencies to scan.
    """
    reqs_pkg = defaultdict(list)
    source = str(absolute_path) if absolute_path else None
    line_locations = line_locations or {}

    for req in dparse_deps:
        reqs_pkg[canonicalize_name(req.name)].append(req)

    for pkg, reqs in reqs_pkg.items():
        specifications_by_key: Dict[tuple, PythonSpecification] = {}

        for req in reqs:
            raw_line = getattr(req, "line", "")
            locations = line_locations.get(raw_line) or []
            location = locations.pop(0) if locations else (source, None)
            line_source, line_number = location

            try:
                specification = parse_requirement(req, line_source)
            except ValueError as exc:
                skipped.append(
                    SkippedRequirement(
                        raw=raw_line or str(req),
                        reason=SkipReason.UNPARSEABLE_LINE,
                        detail=str(exc),
                        source=line_source,
                        line_number=line_number,
                    )
                )
                continue

            original_raw = raw_overrides.get(raw_line)

            if original_raw:
                specification.raw = original_raw

            # Multi-line values (e.g. --hash continuations) are displayed as
            # a single logical line.
            if specification.raw and "\n" in specification.raw:
                specification.raw = " ".join(specification.raw.split())

            if not _marker_applies(specification):
                skipped.append(
                    SkippedRequirement(
                        raw=specification.raw,
                        reason=SkipReason.ENVIRONMENT_MARKER,
                        detail=f"Environment marker '{specification.marker}' "
                        "does not match this environment; pip would not "
                        "install it here.",
                        source=line_source,
                        line_number=line_number,
                    )
                )
                continue

            key = _key_of(specification)

            if key in specifications_by_key:
                # Same package, same effective requirement (e.g. repeated via
                # extras or an included file): merge extras and count once.
                _merge_extras(specifications_by_key[key], specification)
                continue

            # pip resolves all the constraints of a package to a single
            # installed version. When one line pins that version and another
            # only narrows the range, keep the pin as the analyzed
            # requirement, so the vulnerability is reported once instead of
            # once per line (the pin already determines what is installed).
            pinned_match = _matching_pin(specifications_by_key, specification)

            if pinned_match is not None:
                _merge_extras(pinned_match, specification)
                continue

            matching_ranges = _matching_ranges(specifications_by_key, specification)

            if matching_ranges:
                for matching_range in matching_ranges:
                    _merge_extras(specification, matching_range)
                    specifications_by_key.pop(_key_of(matching_range), None)

            specifications_by_key[key] = specification

        specifications = list(specifications_by_key.values())

        if not specifications:
            continue

        version = find_version(specifications)

        yield PythonDependency(
            name=pkg,
            version=version,
            specifications=specifications,
            found=found,
            absolute_path=absolute_path,
            insecure_versions=[],
            secure_versions=[],
            latest_version=None,
            latest_version_without_known_vulnerabilities=None,
            more_info_url=None,
        )


def parse_dependency_content(
    content: str, path: Optional[str], resolve: bool = True
) -> Tuple[List[PythonDependency], List[SkippedRequirement]]:
    """
    Parses dependency file content into the set of dependencies pip would
    install in this environment, plus a traceable list of skipped lines.

    Args:
        content (str): The file content.
        path (Optional[str]): The file path, used to detect the file format.
        resolve (bool): Whether to resolve referenced (``-r``) files.

    Returns:
        Tuple[List[PythonDependency], List[SkippedRequirement]]: The
        dependencies to scan and the skipped entries.
    """
    skipped: List[SkippedRequirement] = []
    raw_overrides: Dict[str, str] = {}
    line_locations: Dict[str, List[Tuple[str, int]]] = {}
    absolute_path: Optional[Path] = Path(path).resolve() if path else None
    found = absolute_path if absolute_path else Path("temp_file")
    name = Path(path).name.lower() if path else ""

    if name == "pipfile.lock":
        normalized = normalize_pipfile_lock(content, str(absolute_path), skipped)
        dparse_deps = _run_dparse(
            normalized, path, filetypes.pipfile_lock, False, skipped
        )
    elif name == "setup.cfg":
        requirement_lines = collect_setup_cfg_lines(
            content, str(absolute_path), skipped
        )
        dparse_deps = _parse_requirement_lines(
            requirement_lines, path, raw_overrides, line_locations, skipped
        )
    elif name == "pipfile":
        requirement_lines = collect_pipfile_lines(content, str(absolute_path), skipped)
        dparse_deps = _parse_requirement_lines(
            requirement_lines, path, raw_overrides, line_locations, skipped
        )
    elif path is None or path.endswith((".txt", ".in")):
        requirement_lines = collect_requirement_lines(
            content, path or "requirements.txt", resolve, skipped, raw_overrides
        )
        dparse_deps = _parse_requirement_lines(
            requirement_lines, path, raw_overrides, line_locations, skipped
        )
    elif name == "pyproject.toml":
        # Only the PEP 508 project dependency groups are installable
        # requirements; build-system requirements are not installed by pip.
        requirement_lines = collect_pyproject_lines(
            content, str(absolute_path), skipped
        )
        dparse_deps = _parse_requirement_lines(
            requirement_lines, path, raw_overrides, line_locations, skipped
        )
    elif path.endswith("poetry.lock"):
        dparse_deps = _run_dparse(
            content, path, filetypes.poetry_lock, resolve, skipped
        )
    else:
        dparse_deps = _run_dparse(content, path, None, resolve, skipped)

    dependencies = list(
        _build_dependencies(
            dparse_deps,
            found,
            absolute_path,
            raw_overrides,
            skipped,
            line_locations,
        )
    )

    return dependencies, skipped


def _parse_requirement_lines(
    requirement_lines: List[Tuple[str, str, int]],
    path: Optional[str],
    raw_overrides: Dict[str, str],
    line_locations: Dict[str, List[Tuple[str, int]]],
    skipped: List[SkippedRequirement],
) -> list:
    """
    Parses already-collected requirement lines with dparse.

    Lines that the parser cannot handle are reported instead of being
    silently dropped.

    Args:
        requirement_lines (List[Tuple[str, str, int]]): (line, source,
            line_number) tuples.
        path (Optional[str]): The file path, used for parser detection.
        raw_overrides (Dict[str, str]): Maps rewritten lines to their original
            raw line.
        line_locations (Dict[str, List[Tuple[str, int]]]): Populated with the
            (source, line number) each parsed line came from.
        skipped (List[SkippedRequirement]): Accumulator for skipped entries.

    Returns:
        list: The parsed dparse dependencies.
    """
    dparse_content = "\n".join(line for line, _, _ in requirement_lines)
    dparse_deps = _run_dparse(
        dparse_content, path, filetypes.requirements_txt, False, skipped
    )
    parsed_lines = {dep.line for dep in dparse_deps}

    for line, line_source, line_number in requirement_lines:
        if line in parsed_lines:
            line_locations.setdefault(line, []).append((line_source, line_number))
        else:
            skipped.append(
                SkippedRequirement(
                    raw=line,
                    reason=SkipReason.UNPARSEABLE_LINE,
                    detail="Not a valid requirement line.",
                    source=line_source,
                    line_number=line_number,
                )
            )

    return dparse_deps


def read_virtual_environment_dependencies(
    f: InspectableFile,
) -> Generator[PythonDependency, None, None]:
    """
    Reads dependencies from a virtual environment.

    Args:
        f (InspectableFile): The inspectable file representing the virtual environment.

    Returns:
        Generator[PythonDependency, None, None]: A generator of PythonDependency objects.
    """

    env_path = Path(f.file.name).resolve().parent

    if sys.platform.startswith("win"):
        site_pkgs_path = env_path / Path("Lib/site-packages/")
    else:
        site_pkgs_path = Path("lib/")
        try:
            site_pkgs_path = next((env_path / site_pkgs_path).glob("*/site-packages/"))
        except StopIteration:
            # Unable to find packages for foo env
            return

    if not site_pkgs_path.resolve().exists():
        # Unable to find packages for foo env
        return

    dep_paths = site_pkgs_path.glob("*/METADATA")

    for path in dep_paths:
        if not path.is_file():
            continue

        dist_info_folder = path.parent
        dep_name, dep_version = dist_info_folder.name.replace(".dist-info", "").split(
            "-"
        )

        yield PythonDependency(
            name=dep_name,
            version=dep_version,
            specifications=[
                PythonSpecification(f"{dep_name}=={dep_version}", found=site_pkgs_path)
            ],
            found=site_pkgs_path,
            insecure_versions=[],
            secure_versions=[],
            latest_version=None,
            latest_version_without_known_vulnerabilities=None,
            more_info_url=None,
        )


def get_dependencies(f: InspectableFile) -> List[PythonDependency]:
    """
    Gets the dependencies for the given inspectable file.

    Any requirement lines that were kept out of the scan set are recorded on
    the inspectable file (``f.skipped_dependencies``) so they stay traceable.

    Args:
        f (InspectableFile): The inspectable file.

    Returns:
        List[PythonDependency]: A list of PythonDependency objects.
    """
    if not f.file_type:
        return []

    if f.file_type in [
        FileType.REQUIREMENTS_TXT,
        FileType.POETRY_LOCK,
        FileType.PIPENV_LOCK,
        FileType.PYPROJECT_TOML,
    ]:
        dependencies, skipped = read_dependencies_with_skipped(f.file, resolve=True)
        f.skipped_dependencies = skipped
        return dependencies

    if f.file_type == FileType.VIRTUAL_ENVIRONMENT:
        return list(read_virtual_environment_dependencies(f))

    return []
