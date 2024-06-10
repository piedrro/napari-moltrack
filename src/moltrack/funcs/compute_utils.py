from qtpy.QtCore import QObject
from qtpy.QtCore import QRunnable
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import traceback
import sys
from multiprocessing import Process, shared_memory, Pool
import numpy as np
import napari

class _compute_utils:

    def create_shared_image_chunks(self, dataset_list = None,
            channel_list = None, chunk_size = 100, frame_index = None):

        if self.verbose:
            print("Creating shared images")

        if dataset_list is None:
            dataset_list = list(self.dataset_dict.keys())
        else:
            dataset_list = [dataset for dataset in dataset_list if dataset in self.dataset_dict.keys()]


        self.shared_chunks = []

        for dataset_name in dataset_list:

            if "images" in self.dataset_dict[dataset_name].keys():

                if channel_list is None:
                    channel_list = list(self.dataset_dict[dataset_name]["images"].keys())

                for channel_name in channel_list:

                    channel_dict = self.dataset_dict[dataset_name]["images"]

                    if type(frame_index) == int:
                        n_chunks = 1
                        image = channel_dict[channel_name]
                        n_frames = 1
                    else:
                        image = channel_dict.pop(channel_name)
                        n_frames = image.shape[0]
                        n_chunks = int(np.ceil(n_frames / chunk_size))

                    for chunk_index in range(n_chunks):

                        if type(frame_index) == int:

                            start_index = frame_index
                            end_index = frame_index + 1

                            chunk = image[start_index].copy()
                            chunk = np.expand_dims(chunk, axis=0)

                        else:
                            start_index = chunk_index * chunk_size
                            end_index = (chunk_index + 1) * chunk_size

                            if end_index > n_frames:
                                end_index = n_frames

                            chunk = image[start_index:end_index]

                        shared_mem = shared_memory.SharedMemory(create=True, size=chunk.nbytes)
                        shared_memory_name = shared_mem.name
                        shared_chunk = np.ndarray(chunk.shape, dtype=chunk.dtype, buffer=shared_mem.buf)
                        shared_chunk[:] = chunk[:]

                        self.shared_chunks.append({"dataset": dataset_name,
                                                   "channel": channel_name,
                                                   "n_frames": n_frames,
                                                   "shape": chunk.shape,
                                                   "dtype": chunk.dtype,
                                                   "start_index": start_index,
                                                   "end_index": end_index,
                                                   "chunk_size": chunk_size,
                                                   "shared_mem": shared_mem,
                                                   "shared_memory_name": shared_memory_name})

    def restore_shared_image_chunks(self):

        if self.verbose:
            print("Restoring shared images")

        if hasattr(self, "shared_chunks"):

            if type(self.shared_chunks) == list:

                dataset_list = []

                for dat in self.shared_chunks:
                    try:
                        dataset = dat["dataset"]
                        channel = dat["channel"]
                        shape = dat["shape"]
                        dtype = dat["dtype"]
                        start_index = dat["start_index"]
                        shared_mem = dat["shared_mem"]

                        np_array = np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf).copy()
                        shared_mem.close()
                        shared_mem.unlink()

                        data_dict = self.dataset_dict[dataset]
                        image_dict = data_dict["images"]

                        if channel not in image_dict.keys():
                            image_dict[channel] = []
                            data_dict["chunk_indices"] = []

                        if type(image_dict[channel]) == list:

                            image_dict[channel].append(np_array)
                            data_dict["chunk_indices"].append(start_index)

                            dataset_list.append(dataset)

                    except:
                        print(traceback.format_exc())
                        pass

                dataset_list = list(set(dataset_list))

                for dataset in dataset_list:

                    try:

                        data_dict = self.dataset_dict[dataset]

                        if "images" in data_dict.keys():

                            image_dict = data_dict["images"]

                            chunk_indices = data_dict["chunk_indices"]
                            sorted_indices = np.argsort(chunk_indices)

                            for channel, images in image_dict.items():

                                if type(images) == list:

                                    sorted_images = [images[i] for i in sorted_indices]
                                    image = np.concatenate(sorted_images, axis=0)

                                    image_dict[channel] = image

                            del data_dict["chunk_indices"]

                    except:
                        print(traceback.format_exc())
                        pass


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
            pass



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

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

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

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:

            while not self._is_stopped:
                result = self.fn(*self.args, **self.kwargs)
                self.signals.result.emit(result)  # Emit the result
                self._is_stopped = True
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
        self.signals.finished.emit()