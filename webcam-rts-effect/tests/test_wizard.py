import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from webcam_effect import cli
from webcam_effect.cli import build_parser
from webcam_effect.components import ComponentSettings
from webcam_effect.wizard import (
    build_dataset_argv,
    build_doctor_argv,
    build_download_argv,
    build_editor_argv,
    build_filters_argv,
    build_run_argv,
)


class WizardBuilderTest(unittest.TestCase):
    def test_run_argv_all_components(self):
        argv = build_run_argv(ComponentSettings(), "0", False, True, "preview")
        self.assertEqual(
            argv,
            [
                "run",
                "--no-components-tui",
                "--components",
                "segment,classify,hand_track",
                "--camera",
                "0",
                "--video-output",
                "preview",
            ],
        )

    def test_run_argv_subset(self):
        argv = build_run_argv(ComponentSettings(classify=False), "0", False, True, "preview")
        self.assertIn("segment,hand_track", argv)
        self.assertNotIn("--debug", argv)
        self.assertNotIn("--no-audio", argv)

    def test_run_argv_debug_and_no_audio(self):
        argv = build_run_argv(ComponentSettings(), "0", True, False, "none")
        self.assertIn("--debug", argv)
        self.assertIn("--no-audio", argv)
        self.assertEqual(argv[argv.index("--video-output") + 1], "none")

    def test_filters_argv(self):
        self.assertEqual(
            build_filters_argv("1", "1280x720", "none"),
            ["filters", "--camera", "1", "--resolution", "1280x720", "--video-output", "none"],
        )

    def test_dataset_argv(self):
        self.assertEqual(build_dataset_argv("kicau", "0"), ["dataset", "--label", "kicau", "--camera", "0"])

    def test_simple_builders(self):
        self.assertEqual(build_editor_argv(), ["editor"])
        self.assertEqual(build_doctor_argv(), ["doctor"])
        self.assertEqual(build_download_argv(), ["download-models"])


class WizardParserRoundTripTest(unittest.TestCase):
    def test_run_argv_round_trips(self):
        args = build_parser().parse_args(build_run_argv(ComponentSettings(), "1", True, False, "none"))
        self.assertEqual(args.command, "run")
        self.assertEqual(args.camera, "1")
        self.assertTrue(args.debug)
        self.assertTrue(args.no_audio)
        self.assertFalse(args.components_tui)
        self.assertEqual(args.video_output, "none")

    def test_filters_argv_round_trips(self):
        args = build_parser().parse_args(build_filters_argv("2", "1280x720", "none"))
        self.assertEqual(args.command, "filters")
        self.assertEqual(args.camera, "2")
        self.assertEqual(args.resolution, "1280x720")
        self.assertEqual(args.video_output, "none")

    def test_dataset_argv_round_trips(self):
        args = build_parser().parse_args(build_dataset_argv("kicau", "0"))
        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.label, "kicau")
        self.assertEqual(args.camera, "0")


class BareInvocationTest(unittest.TestCase):
    def test_bare_non_tty_prints_help(self):
        buffer = io.StringIO()
        with mock.patch.object(cli.sys.stdin, "isatty", return_value=False):
            with redirect_stdout(buffer):
                cli.main([])
        self.assertIn("usage:", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
