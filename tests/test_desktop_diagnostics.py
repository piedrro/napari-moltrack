"""Exercise the computations used to check a frozen build."""


def test_scientific_probe_reads_images_and_compiles_numba():
    from moltrack.desktop_diagnostics import scientific_probe

    result = scientific_probe()
    assert result["image_sum"] == 120
    assert result["compiled_sum"] == 120
    assert result["cpu_tensor_sum"] == 120
    assert result["pid"] > 0


def test_packaged_checks_spawn_workers_and_write_report(tmp_path, qtbot):
    import json
    import os

    from moltrack.desktop_diagnostics import run_checks
    from moltrack.launcher import create_viewer

    viewer = create_viewer(show=False)
    try:
        report = tmp_path / "smoke.json"
        run_checks(viewer, report)
        result = json.loads(report.read_text(encoding="utf-8"))
        assert result["ok"] is True
        assert result["worker"]["pid"] != os.getpid()
        assert result["layers"] == 0
        assert result["manager_value"] == "ready"
        assert result["plugins"] == ["napari", "napari-moltrack"]
    finally:
        viewer.close()
