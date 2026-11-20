import unittest
import tempfile
from importlib.metadata import version
from click.testing import CliRunner
from packaging.version import Version
from safety.auth.models import Auth
from safety.cli import cli
from safety.console import main_console as console
from unittest.mock import patch


class TestScanCommand(unittest.TestCase):
    def setUp(self):
        # mix_stderr was removed in Click 8.2.0
        if Version(version("click")) >= Version("8.2.0"):
            self.runner = CliRunner()
        else:
            self.runner = CliRunner(mix_stderr=False)
        self.target = tempfile.mkdtemp()
        # Make sure the console is not quiet
        # TODO: This is a workaround, we should improve the way the console
        # is initialized in the CLI
        console.quiet = False

        cli.commands = cli.all_commands
        self.cli = cli

        # Prevent real machine credentials from leaking into tests.
        # AUTH_CONFIG_USER is a module-level constant resolved at import time,
        # so patching get_user_dir does NOT redirect it.
        self.machine_creds_patcher = patch(
            "safety.auth.cli_utils.MachineCredentialConfig.from_storage",
            return_value=None,
        )
        self.machine_creds_patcher.start()

    def tearDown(self):
        self.machine_creds_patcher.stop()

    @patch.object(Auth, "is_valid", return_value=False)
    @patch(
        "safety.platform.SafetyPlatformClient.get_authentication_type",
        return_value="unauthenticated",
    )
    def test_scan(self, mock_is_valid, mock_get_auth_type):
        result = self.runner.invoke(
            self.cli, ["scan", "--target", self.target, "--output", "json"]
        )
        self.assertEqual(result.exit_code, 1)

        result = self.runner.invoke(
            self.cli,
            [
                "--stage",
                "production",
                "scan",
                "--target",
                self.target,
                "--output",
                "json",
            ],
        )
        self.assertEqual(result.exit_code, 1)

        result = self.runner.invoke(
            self.cli,
            ["--stage", "cicd", "scan", "--target", self.target, "--output", "screen"],
        )
        self.assertEqual(result.exit_code, 1)


class TestSkippedDependenciesReport(unittest.TestCase):
    def test_adds_skipped_dependencies_to_matching_file(self):
        import json
        from pathlib import Path
        from safety_schemas.models import FileType
        from safety_schemas.models.file import FileModel
        from safety.scan.command import add_skipped_dependencies_to_report

        location = Path("/tmp/project/requirements.txt")
        file_model = FileModel(location=location, file_type=FileType.REQUIREMENTS_TXT)
        file_model.skipped_dependencies = [
            {
                "raw": 'django==1.6 ; python_version < "3"',
                "reason": "environment-marker",
                "detail": "marker does not match",
                "source": str(location),
                "line_number": 1,
            }
        ]

        report = json.dumps(
            {
                "scan_results": {
                    "files": [{"location": str(location), "results": {}}],
                    "projects": [
                        {
                            "files": [
                                {"location": str(location), "results": {}},
                                {"location": "/tmp/project/other.txt", "results": {}},
                            ]
                        }
                    ],
                }
            }
        )

        updated = json.loads(add_skipped_dependencies_to_report(report, [file_model]))

        # Both the top-level and the nested project file entry carry the skip.
        self.assertEqual(
            updated["scan_results"]["files"][0]["results"]["skipped_dependencies"][0][
                "reason"
            ],
            "environment-marker",
        )
        project_files = updated["scan_results"]["projects"][0]["files"]
        self.assertEqual(len(project_files[0]["results"]["skipped_dependencies"]), 1)
        self.assertNotIn("skipped_dependencies", project_files[1]["results"])

    def test_report_is_unchanged_when_nothing_was_skipped(self):
        from pathlib import Path
        from safety_schemas.models import FileType
        from safety_schemas.models.file import FileModel
        from safety.scan.command import add_skipped_dependencies_to_report

        file_model = FileModel(
            location=Path("/tmp/project/requirements.txt"),
            file_type=FileType.REQUIREMENTS_TXT,
        )
        report = '{"scan_results": {"projects": []}}'

        self.assertEqual(
            add_skipped_dependencies_to_report(report, [file_model]), report
        )

    def test_malformed_report_is_returned_unchanged(self):
        from pathlib import Path
        from safety_schemas.models import FileType
        from safety_schemas.models.file import FileModel
        from safety.scan.command import add_skipped_dependencies_to_report

        file_model = FileModel(
            location=Path("/tmp/project/requirements.txt"),
            file_type=FileType.REQUIREMENTS_TXT,
        )
        file_model.skipped_dependencies = [{"raw": "x", "reason": "vcs-dependency"}]

        self.assertEqual(
            add_skipped_dependencies_to_report("not json", [file_model]), "not json"
        )


class TestFixableRequirementsFile(unittest.TestCase):
    def test_only_txt_requirement_files_are_auto_fixable(self):
        from pathlib import Path
        from safety.scan.command import is_fixable_requirements_file

        self.assertTrue(is_fixable_requirements_file(Path("/repo/requirements.txt")))
        self.assertTrue(is_fixable_requirements_file(Path("/repo/REQUIREMENTS.TXT")))
        self.assertFalse(is_fixable_requirements_file(Path("/repo/requirements.in")))
        self.assertFalse(is_fixable_requirements_file(Path("/repo/setup.cfg")))
        self.assertFalse(is_fixable_requirements_file(Path("/repo/Pipfile")))
