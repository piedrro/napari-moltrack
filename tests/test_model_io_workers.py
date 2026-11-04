"""I/O workers receive detached payloads and never query live GUI state."""

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import tifffile

from moltrack.funcs.export_utils import _export_utils
from moltrack.funcs.import_utils import _import_utils
from moltrack.model_io import import_images
from moltrack.model_views import gui_query
from moltrack.session import ProjectSession


def test_queued_image_import_keeps_captured_clear_and_concatenate(monkeypatch, tmp_path):
    from moltrack.funcs import import_utils

    path = tmp_path / "image.tif"
    tifffile.imwrite(path, np.ones((2, 8, 8), dtype=np.uint16))
    host = _import_utils()
    original = host.session = ProjectSession()
    switches = {"clear": False, "concatenate": False}
    host.gui = NS(
        import_clear=NS(isChecked=lambda: switches["clear"]),
        import_concatenate=NS(isChecked=lambda: switches["concatenate"]),
        import_progressbar=None,
    )
    host.populate_import_jobs = lambda **kw: [{
        "path": str(path), "dataset_name": "captured", "pixel_size": 100., "exposure_time": 20.,
        "multichannel_mode": "None", "channel_name": "image", "import_limit": "None",
    }]
    # Actual image preparation, with process scheduling replaced by a direct call.
    monkeypatch.setattr(import_utils, "process_compute_jobs", lambda jobs, *args:
                        [import_utils.import_image_data(job) for job in jobs])
    host.verbose = False
    host.update_ui = lambda **kwargs: None
    host.import_data_finished = lambda: None
    host.moltrack_progress = lambda *args, **kwargs: None
    workers = []
    host.threadpool = NS(start=workers.append)
    host.init_import_data(import_mode="data", import_path=str(path))
    assert len(workers) == 1
    assert getattr(workers[0].fn, "__self__", None) is None
    switches.update(clear=True, concatenate=True)
    host.publish_image_import(workers[0].result())
    assert host.session is original
    np.testing.assert_array_equal(host.session.image("captured", "image")[:], 1)


def test_table_export_worker_uses_captured_rows_and_path(tmp_path):
    class Host(_export_utils):
        def get_locs(self, dataset, channel, **kwargs):
            return gui_query(self.session, dataset, channel, **kwargs)

    host = Host()
    host.session = ProjectSession()
    import_images(host.session, [{
        "path": str(tmp_path / "camera.tif"), "dataset_name": "camera", "pixel_size": 100.,
        "exposure_time": 20., "source_images": np.zeros((2, 8, 8)),
        "multichannel_mode": "None", "channel_name": "image",
    }])
    host.session.replace_localizations("camera", "image", pd.DataFrame({
        "frame": [0], "x": [2.], "y": [3.], "custom": [42.],
    }))
    def control(value):
        return NS(currentText=lambda: value)
    host.gui = NS(locs_export_dataset=control("camera"), locs_export_channel=control("image"),
        locs_export_data=control("Localisations"), locs_export_mode=control("CSV"),
        locs_export_concat=NS(isChecked=lambda: False), export_progressbar=None)
    host.update_ui = lambda **kwargs: None
    host.moltrack_progress = lambda *args, **kwargs: None
    workers = []
    host.threadpool = NS(start=workers.append)
    host.initialise_export_locs()
    assert len(workers) == 1
    assert getattr(workers[0].fn, "__self__", None) is None
    host.session.delete_rows("camera", "image")
    host.session.update_metadata("camera", path=str(tmp_path / "wrong.tif"))
    workers[0].result()
    data = pd.read_csv(tmp_path / "camera_image_moltrack_localisations.csv")
    assert data.x.tolist() == [2.]
    assert data.custom.tolist() == [42.]
