import traceback

import pandas as pd

from moltrack.funcs.compute_utils import Worker
from picasso.render import render
import numpy as np
import time
from functools import partial

class _picasso_render_utils:


    def render_picasso_locs(self, loc_list, image_shape, blur_method=None, min_blur_width=1,
            pixel_size=1, progress_callback=None, oversampling=20, ):
        try:

            h, w = image_shape[-2:]

            viewport = [(float(0), float(0)),
                        (float(h), float(w))]

            start_time = time.time()

            images = []
            total_rendered_locs = 0

            print(f"Rendering localisations from {len(loc_list)} dataset(s)/channel(s).")

            for locs in loc_list:

                n_rendered_locs, image = render(locs,
                    viewport=viewport,
                    blur_method=blur_method,
                    min_blur_width=min_blur_width,
                    oversampling=oversampling,
                    ang=0,
                )

                images.append(image)
                total_rendered_locs += n_rendered_locs

            image = images[0]

            end_time = time.time()

            print(f"Rendered {total_rendered_locs} localisations in {end_time - start_time:.2f} seconds.")

        except:
            print(traceback.format_exc())
            image = np.zeros(image_shape[-2:], dtype=np.int8)

        return image, pixel_size, oversampling

    def picasso_render_finished(self):

        self.update_filter_criterion()
        self.update_criterion_ranges()

        self.update_ui(init=False)


    def draw_picasso_render(self, data):

        try:
            image, pixel_size, oversampling = data

            scale = [pixel_size / oversampling, pixel_size / oversampling]

            layer_names = [layer.name for layer in self.viewer.layers]

            if "SMLM Render" not in layer_names:
                self.viewer.add_image(image, name="SMLM Render", colormap="viridis", scale=scale, )
            else:
                self.viewer.layers["SMLM Render"].data = image
                self.viewer.layers["SMLM Render"].scale = scale

        except:
            print(traceback.format_exc())

    def initialise_picasso_render(self):

        try:

            dataset = self.gui.picasso_render_dataset.currentText()
            channel = self.gui.picasso_render_channel.currentText()
            blur_method = self.gui.picasso_render_blur_method.currentText()
            min_blur_width = float(self.gui.picasso_render_min_blur.text())

            if dataset == "All Datasets":
                dataset_list = list(self.localisation_dict.keys())
            else:
                dataset_list = [dataset]

            if channel == "All Channels":
                channel_list = []
                for dataset_name in self.dataset_dict.keys():
                    try:
                        image_dict = self.dataset_dict[dataset_name]["images"]
                        channel_list.append(set(image_dict.keys()))
                    except:
                        pass

                channel_list = set.intersection(*channel_list)
                channel_list = list(channel_list)

            else:
                channel_list = [channel]


            loc_list = []

            for dataset in dataset_list:

                if channel in self.localisation_dict[dataset].keys():
                    loc_dict = self.localisation_dict[dataset][channel]

                    if "localisations" in loc_dict.keys():
                        loc_list.append(loc_dict["localisations"].copy())

            if len(loc_list) > 0:

                image_dict = self.dataset_dict[dataset_list[0]]["images"]

                image_shape = list(image_dict[channel_list[0]].shape)
                pixel_size = 1

                if blur_method == "One-Pixel-Blur":
                    blur_method = "smooth"
                elif blur_method == "Global Localisation Precision":
                    blur_method = "convolve"
                elif (blur_method == "Individual Localisation Precision, iso"):
                    blur_method = "gaussian_iso"
                elif blur_method == "Individual Localisation Precision":
                    blur_method = "gaussian"
                else:
                    blur_method = None

                self.update_ui(init=True)

                worker = Worker(self.render_picasso_locs, loc_list=loc_list, image_shape=image_shape,
                    blur_method=blur_method, min_blur_width=min_blur_width, pixel_size=pixel_size)
                worker.signals.result.connect(self.draw_picasso_render)
                worker.signals.finished.connect(self.picasso_render_finished)
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()