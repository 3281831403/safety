# type: ignore
"""
Acceptance tests: the scan set must match pip install semantics, and every
skipped or ignored requirement must be traceable in the report.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from safety_schemas.models import ConfigModel, Ecosystem, FileType, IgnoreCodes

from safety.constants import IGNORE_UNPINNED_REQ_REASON
from safety.scan.ecosystems.base import SkipReason
from safety.scan.ecosystems.python.dependencies import parse_dependency_content
from safety.scan.ecosystems.python.main import PythonFile
from safety.scan.finder.file_finder import FileFinder

TEST_DB_DIR = Path(__file__).resolve().parents[3] / "test_db"


def load_test_db(db_name: str, **kwargs):
    with open(TEST_DB_DIR / db_name) as db_file:
        return json.load(db_file)


def write(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class EnvironmentMarkerTestCase(unittest.TestCase):
    """Acceptance 1: markers that do not apply here are not scanned."""

    def test_marker_not_matching_is_skipped(self):
        content = (
            'django==1.6 ; python_version < "3"\nflask==2.0.0 ; python_version >= "3"\n'
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual([dependency.name for dependency in dependencies], ["flask"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.ENVIRONMENT_MARKER)
        self.assertIn("django==1.6", skipped[0].raw)
        self.assertIn('python_version < "3"', skipped[0].detail)
        # The skipped line is traceable back to its source line.
        self.assertEqual(skipped[0].source, str(path.resolve()))
        self.assertEqual(skipped[0].line_number, 1)


class DuplicateRequirementTestCase(unittest.TestCase):
    """Acceptance 2: the same package repeated is counted once."""

    def test_duplicate_specifications_are_counted_once(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write(tmp_path, "extra.txt", "django==1.6\n")
            main = write(
                tmp_path,
                "requirements.txt",
                "django==1.6\ndjango[argon2]==1.6\n-r extra.txt\n",
            )

            dependencies, skipped = parse_dependency_content(
                main.read_text(), str(main), resolve=True
            )

        self.assertEqual(len(dependencies), 1)
        dependency = dependencies[0]
        self.assertEqual(dependency.name, "django")
        self.assertEqual(dependency.version, "1.6")
        self.assertEqual(len(dependency.specifications), 1)
        # extras are merged into the surviving specification
        self.assertEqual(dependency.specifications[0].extras, {"argon2"})
        self.assertEqual(skipped, [])

    def test_pin_and_range_of_the_same_package_count_once(self):
        """A pin plus a range resolve to the pinned version, like pip does."""

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = write(
                tmp_path,
                "requirements.txt",
                "django==1.6\ndjango>=1.0\n",
            )

            dependencies, skipped = parse_dependency_content(
                main.read_text(), str(main), resolve=True
            )

        self.assertEqual(len(dependencies), 1)
        dependency = dependencies[0]
        self.assertEqual(dependency.version, "1.6")
        self.assertEqual(len(dependency.specifications), 1)
        self.assertEqual(str(dependency.specifications[0].specifier), "==1.6")
        self.assertEqual(skipped, [])

    def test_pin_and_incompatible_range_are_kept_separate(self):
        """Two lines pip cannot satisfy together are not silently merged."""

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = write(
                tmp_path,
                "requirements.txt",
                "django==1.6\ndjango>=2.0\n",
            )

            dependencies, _ = parse_dependency_content(
                main.read_text(), str(main), resolve=True
            )

        self.assertEqual(len(dependencies), 1)
        self.assertEqual(
            sorted(
                str(specification.specifier)
                for specification in dependencies[0].specifications
            ),
            ["==1.6", ">=2.0"],
        )

    def test_markers_are_evaluated_without_merging_across_them(self):
        """
        Markers that both apply are kept as distinct requirements when they
        describe different environments.
        """

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            main = write(
                tmp_path,
                "requirements.txt",
                'django==1.6 ; python_version >= "3"\ndjango>=1.0 ; sys_platform == "linux"\n',
            )

            dependencies, skipped = parse_dependency_content(
                main.read_text(), str(main), resolve=True
            )

        self.assertEqual(len(dependencies), 1)
        self.assertEqual(len(dependencies[0].specifications), 2)
        self.assertEqual(skipped, [])


class UnanalyzableDependencyTestCase(unittest.TestCase):
    """Acceptance 3: VCS/editable/local path lines are reported, not dropped."""

    def test_vcs_editable_and_local_path_are_reported_skipped(self):
        content = (
            "git+https://github.com/pallets/flask.git@2.0.0#egg=flask\n"
            "-e git+https://github.com/jdunck/python-unicodecsv#egg=unicodecsv\n"
            "./local_pkg\n"
            "requests==2.31.0\n"
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual([dependency.name for dependency in dependencies], ["requests"])
        self.assertEqual(
            [item.reason for item in skipped],
            [
                SkipReason.VCS_DEPENDENCY,
                SkipReason.EDITABLE_DEPENDENCY,
                SkipReason.LOCAL_PATH_DEPENDENCY,
            ],
        )
        for item in skipped:
            self.assertTrue(item.detail)
            self.assertTrue(item.raw)

    def test_unresolved_include_is_reported(self):
        content = "-r does-not-exist.txt\nrequests==2.31.0\n"

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual([dependency.name for dependency in dependencies], ["requests"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.UNRESOLVED_INCLUDE)


class DirectReferenceTestCase(unittest.TestCase):
    """Acceptance 4: PEP 508 direct references are pinned from the URL."""

    def test_pep508_direct_reference_is_pinned_from_url(self):
        content = (
            "django @ https://example.com/packages/django-1.6.tar.gz\n"
            "flask @ https://example.com/download?id=42\n"
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual(len(dependencies), 1)
        django = dependencies[0]
        self.assertEqual(django.name, "django")
        self.assertEqual(django.version, "1.6")
        specification = django.specifications[0]
        self.assertEqual(str(specification.specifier), "==1.6")
        self.assertTrue(specification.is_pinned())
        # the raw line keeps the original PEP 508 form for traceability
        self.assertEqual(
            specification.raw,
            "django @ https://example.com/packages/django-1.6.tar.gz",
        )

        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.UNRESOLVABLE_DIRECT_REFERENCE)
        self.assertIn("flask", skipped[0].raw)

    def test_bare_archive_url_is_pinned_from_egg_fragment(self):
        content = "https://example.com/flask-2.0.0.tar.gz#egg=flask\n"

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual(skipped, [])
        self.assertEqual(
            [(dependency.name, dependency.version) for dependency in dependencies],
            [("flask", "2.0.0")],
        )


class PyprojectTestCase(unittest.TestCase):
    """pyproject.toml follows the same scan set and traceability rules."""

    def test_project_dependencies_follow_pip_semantics(self):
        content = (
            '[project]\nname = "demo"\ndependencies = [\n'
            '    "django==1.6",\n'
            "    'flask>=2.0 ; python_version < \"3\"',\n"
            "]\n"
            "\n[project.optional-dependencies]\n"
            'dev = ["pytest==7.0.0"]\n'
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "pyproject.toml", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual(
            sorted(
                (dependency.name, dependency.version) for dependency in dependencies
            ),
            [("django", "1.6"), ("pytest", "7.0.0")],
        )
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.ENVIRONMENT_MARKER)
        self.assertEqual(skipped[0].line_number, 5)


class HashContinuationTestCase(unittest.TestCase):
    """Acceptance 5: multi-line --hash pins are a single logical line."""

    def test_multiline_hash_is_a_single_logical_line(self):
        content = (
            "django==1.6 \\\n"
            "    --hash=sha256:" + "a" * 64 + " \\\n"
            "    --hash=sha256:" + "b" * 64 + "\n"
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual(skipped, [])
        self.assertEqual(len(dependencies), 1)
        specification = dependencies[0].specifications[0]
        self.assertEqual(dependencies[0].version, "1.6")
        self.assertNotIn("\n", specification.raw)
        self.assertNotIn("\\", specification.raw)
        self.assertEqual(specification.raw.count("--hash="), 2)


class PipfileLockTestCase(unittest.TestCase):
    """Acceptance 6: a Pipfile.lock entry without hashes does not crash."""

    def test_entry_without_hashes_does_not_crash(self):
        content = json.dumps(
            {
                "_meta": {"hash": {"sha256": "abc"}, "pipfile-spec": 6},
                "default": {
                    "django": {"version": "==1.6"},
                    "requests": {
                        "hashes": ["sha256:abc"],
                        "version": "==2.31.0",
                    },
                    "local-tool": {"git": "https://github.com/x/y.git"},
                },
                "develop": {},
            }
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "Pipfile.lock", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual(sorted(d.name for d in dependencies), ["django", "requests"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.VCS_DEPENDENCY)
        self.assertIn("local-tool", skipped[0].raw)

    def test_malformed_file_is_reported_not_raised(self):
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "Pipfile.lock", "{not json")
            dependencies, skipped = parse_dependency_content(
                "{not json", str(path), resolve=True
            )

        self.assertEqual(dependencies, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.FILE_PARSE_ERROR)


@patch(
    "safety.scan.ecosystems.python.main.get_from_cache",
    side_effect=load_test_db,
)
class InspectSemanticsTestCase(unittest.TestCase):
    def inspect(self, path: Path, file_type=FileType.REQUIREMENTS_TXT) -> PythonFile:
        with open(path, "r+") as file:
            python_file = PythonFile(file_type=file_type, file=file)
            python_file.inspect(config=ConfigModel())
            python_file.remediate()
        return python_file

    def test_unpinned_specification_is_ignored_with_a_reason(self, _mock_cache):
        """
        Acceptance 7: an unpinned requirement keeps its vulnerabilities but
        they are ignored with a traceable reason.
        """
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", "django>=1.0\n")
            python_file = self.inspect(path)

        results = python_file.dependency_results
        self.assertEqual(len(results.dependencies), 1)

        specification = results.dependencies[0].specifications[0]
        self.assertGreater(len(specification.vulnerabilities), 0)

        for vulnerability in specification.vulnerabilities:
            self.assertTrue(vulnerability.ignored)
            self.assertEqual(
                vulnerability.ignored_code, IgnoreCodes.unpinned_specification.value
            )
            self.assertEqual(vulnerability.ignored_reason, IGNORE_UNPINNED_REQ_REASON)

        self.assertTrue(results.ignored_vulns_data)

    def test_duplicate_django_reports_vulnerabilities_once(self, _mock_cache):
        """
        Acceptance 2 (end to end): duplicated pins of the same package yield
        one affected specification and each vulnerability is counted once.
        """
        content = "django==1.6\ndjango[argon2]==1.6\n"

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            python_file = self.inspect(path)

        results = python_file.dependency_results
        affected = results.get_affected_specifications()
        self.assertEqual(len(affected), 1)

        vulnerabilities = [
            vuln for vuln in affected[0].vulnerabilities if not vuln.ignored
        ]
        self.assertEqual(len(vulnerabilities), 2)

    def test_marker_excluded_dependency_reports_zero_dependencies(self, _mock_cache):
        """
        Acceptance 1 (end to end): a file whose only requirement is excluded
        by its environment marker scans zero dependencies and records the skip.
        """
        with TemporaryDirectory() as tmp:
            path = write(
                Path(tmp), "requirements.txt", 'django==1.6 ; python_version < "3"\n'
            )
            python_file = self.inspect(path)

        self.assertEqual(python_file.dependency_results.dependencies, [])
        self.assertEqual(len(python_file.skipped_dependencies), 1)
        self.assertEqual(
            python_file.skipped_dependencies[0].reason,
            SkipReason.ENVIRONMENT_MARKER,
        )

    def test_pin_plus_range_reports_each_vulnerability_once(self, _mock_cache):
        """
        Acceptance 2 (end to end): ``django==1.6`` and ``django>=1.0`` are the
        same installed version, so the 2 vulnerabilities are reported once and
        produce a single fix - not 4 vulnerabilities and 2 fixes.
        """
        content = "django==1.6\ndjango>=1.0\n"

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "requirements.txt", content)
            python_file = self.inspect(path)

        results = python_file.dependency_results
        affected = results.get_affected_specifications()
        self.assertEqual(len(affected), 1)

        vulnerabilities = [
            vuln for vuln in affected[0].vulnerabilities if not vuln.ignored
        ]
        self.assertEqual(len(vulnerabilities), 2)
        self.assertIsNotNone(affected[0].remediation)
        self.assertTrue(affected[0].remediation.recommended)


class FileDiscoveryTestCase(unittest.TestCase):
    """Acceptance 8: setup.cfg/Pipfile/requirements.in are discovered."""

    def test_discovers_setup_cfg_pipfile_and_requirements_in(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp)
            write(
                target,
                "setup.cfg",
                "[options]\ninstall_requires =\n    django==1.6\n",
            )
            write(target, "Pipfile", '[packages]\ndjango = "==1.6"\n')
            write(target, "requirements.in", "django==1.6\n")

            finder = FileFinder(
                max_level=3, ecosystems=[Ecosystem.PYTHON], target=target
            )
            _, files = finder.search()

            discovered = {path.name for paths in files.values() for path in paths}
            self.assertEqual(discovered, {"setup.cfg", "Pipfile", "requirements.in"})

            for name in ("setup.cfg", "Pipfile", "requirements.in"):
                path = target / name
                dependencies, _ = parse_dependency_content(
                    path.read_text(), str(path), resolve=True
                )
                self.assertEqual(
                    [dependency.name for dependency in dependencies],
                    ["django"],
                    f"{name} should yield the django dependency",
                )

    def test_unrelated_in_templates_are_not_discovered(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp)
            write(target, "Makefile.in", "all:\n\t@echo build\n")
            write(target, "requirements.in", "django==1.6\n")

            finder = FileFinder(
                max_level=3, ecosystems=[Ecosystem.PYTHON], target=target
            )
            _, files = finder.search()

            discovered = {path.name for paths in files.values() for path in paths}
            self.assertEqual(discovered, {"requirements.in"})

    def test_pipfile_markers_follow_pip_semantics(self):
        content = (
            "[packages]\n"
            'django = {version = "==1.6", markers = "python_version < \'3\'"}\n'
            'flask = {version = "==2.0.0", markers = "python_version >= \'3\'"}\n'
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "Pipfile", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual(
            [(dependency.name, dependency.version) for dependency in dependencies],
            [("flask", "2.0.0")],
        )
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.ENVIRONMENT_MARKER)

    def test_poetry_pyproject_is_reported_not_scanned_as_empty(self):
        content = (
            '[tool.poetry]\nname = "demo"\n[tool.poetry.dependencies]\ndjango = "1.6"\n'
        )

        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "pyproject.toml", content)
            dependencies, skipped = parse_dependency_content(
                content, str(path), resolve=True
            )

        self.assertEqual(dependencies, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0].reason, SkipReason.UNSUPPORTED_FORMAT)


if __name__ == "__main__":
    unittest.main()
