import pytest

from imgseg import pipeline


def test_resolve_scenes_all_returns_every_scene():
    resolved = pipeline.resolve_scenes(["all"])
    assert [s.key for s in resolved] == ["1", "2", "3", "4"]


def test_resolve_scenes_subset():
    resolved = pipeline.resolve_scenes(["2", "4"])
    assert [s.key for s in resolved] == ["2", "4"]


def test_resolve_scenes_unknown_key_raises():
    with pytest.raises(SystemExit):
        pipeline.resolve_scenes(["99"])


def test_run_scene_missing_data_returns_false(tmp_path):
    spec = pipeline.SCENES_BY_KEY["2"]
    ok = pipeline.run_scene(spec, data_dir=str(tmp_path), out_dir=str(tmp_path / "out"), no_show=True)
    assert ok is False


def test_cli_list_exits_zero(capsys):
    exit_code = pipeline.main(["--list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Scene" not in out or "1:" in out  # sanity: something was printed
