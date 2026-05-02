"""Smoke tests for the dmr CLI."""
import json
import subprocess
import sys

import pytest

from classifier.cli import main


def test_presets_subcommand(capsys):
    rc = main(["presets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "healthcare" in out
    assert "legal" in out
    assert "fintech" in out


def test_classify_subcommand(capsys):
    rc = main(["classify", "What is 2+2?"])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert body["tier"] in ("low", "medium", "high")
    assert body["model"]
    assert body["task_type"]


def test_classify_with_preset(capsys):
    rc = main(["classify", "Patient MRN: 12345678 has elevated AST",
               "--preset", "healthcare"])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert body["compliance_flag"] is True


def test_init_writes_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main(["init"])
    assert rc == 0
    cfg = tmp_path / "dmr.yaml"
    assert cfg.exists()
    text = cfg.read_text(encoding="utf-8")
    assert "providers:" in text
    assert "layer3_enabled" in text


def test_init_refuses_overwrite_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dmr.yaml").write_text("existing", encoding="utf-8")
    rc = main(["init"])
    assert rc == 1
    assert (tmp_path / "dmr.yaml").read_text() == "existing"


def test_init_force_overwrites(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dmr.yaml").write_text("old content", encoding="utf-8")
    rc = main(["init", "--force"])
    assert rc == 0
    assert "providers:" in (tmp_path / "dmr.yaml").read_text(encoding="utf-8")


def test_help_shows_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for sub in ("classify", "train", "stats", "init", "presets"):
        assert sub in out
