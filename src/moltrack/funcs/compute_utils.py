import sys
import traceback
from concurrent.futures import CancelledError

import napari
import numpy as np
from qtpy.QtCore import Signal, Slot
from qtpy.QtCore import QObject, QRunnable


def publish_model_result(host, pending):
    """GUI result boundary shared by scientific processing controls."""
    from concurrent.futures import CancelledError
    from napari.utils.notifications import show_info
    from moltrack.model_processing import publish
    try:
        publish(host.session, pending, cancel_event=host.stop_event)
        return True
    except CancelledError:
        return False
    except Exception as exc:
        show_info(f"Processing result was not applied: {exc}")
        return False


class _compute_utils:




    def clear_live_images(self):

        try:

            if self.verbose:
                print("Clearing live images")

            image_layers = [layer for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]

            for layer in image_layers:

                frame_shape = layer.data.shape[1:]
                empty_frame = np.zeros(frame_shape, dtype=layer.data.dtype)
                layer.data = empty_frame

        except:
            print(traceback.format_exc())



class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    """

    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)

class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

        self._is_stopped = False  # Stop flag

    @Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:

            if not self._is_stopped:
                result = self.fn(*self.args, **self.kwargs)
                if not self._is_stopped:
                    self.signals.result.emit(result)
        except CancelledError:
            pass
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()  # Done

    def result(self):
        return self.fn(*self.args, **self.kwargs)

    def stop(self):

        self._is_stopped = True
