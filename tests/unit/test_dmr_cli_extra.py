"""Tests for the new dmr subcommands: doctor, version, benchmark."""

import json
import sys
from io import StringIO
from unittest.mock import patch

from classifier.cli import main


def _run(argv) -> tuple[int, str]:
    """Run dmr with argv, capture stdout, return (exit_code, stdout)."""
    captured = StringIO()
    with patch.object(sys, "stdout", captured):
        rc = main(argv)
    return rc, captured.getvalue()


def test_version_subcommand_outputs_json():
    rc, out = _run(["version"])
    assert rc == 0
    parsed = json.loads(out)
    assert "dynamic_model_router" in parsed
    assert "python" in parsed
    assert parsed["dynamic_model_router"]


def test_doctor_subcommand_runs():
    rc, out = _run(["doctor"])
    assert rc in (0, 1)  # 1 if any FAIL check
    assert "Result:" in out
    assert "Python version" in out


def test_benchmark_subcommand_runs():
    rc, out = _run(["benchmark", "--iterations", "2"])
    assert rc == 0
    assert "p50" in out
    assert "p95" in out


def test_help_lists_new_subcommands():
    captured = StringIO()
    with patch.object(sys, "stdout", captured):
        try:
            main(["--help"])
        except SystemExit:
            pass  # argparse exits 0 on --help
    out = captured.getvalue()
    assert "doctor" in out
    assert "version" in out
    assert "benchmark" in out
    assert "eval" in out
