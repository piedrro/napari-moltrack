import traceback

import pandas as pd

from moltrack.funcs.compute_utils import Worker
from picasso.render import render
import numpy as np
import time
from functools import partial
import cv2
from napari.utils.notifications import show_info


class _picasso_render_utils:

    def render_picasso_locs(
        self,
        loc_data,
        image_shape,
        blur_method=None,
        min_blur_width=1,
        pixel_size=1,
        progress_callback=None,
        oversampling=10,
    ):
        try:

            h, w = image_shape[-2:]

            viewport = [(float(0), float(0)), (float(h), float(w))]

            start_time = time.time()

            images = []
            total_rendered_locs = 0

            show_info(f"Rendering localisations from {len(loc_data)} dataset(s)/channel(s).")

            for dat in loc_data:

                locs = dat["localisations"]

                n_rendered_locs, image = render(
                    locs,
                    viewport=viewport,
                    blur_method=blur_method,
                    min_blur_width=min_blur_width,
                    oversampling=oversampling,
                    ang=0,
                )

                images.append(image)
                total_rendered_locs += n_rendered_locs

            if len(images) == 0:
                image = np.zeros(image_shape[-2:], dtype=np.int8)
            else:
                image = self.create_rgb_render(images)

            end_time = time.time()

            show_info(f"Rendered {total_rendered_locs} localisations in {end_time - start_time:.2f} seconds.")

        except:
            print(traceback.format_exc())
            image = np.zeros(image_shape[-2:], dtype=np.int8)

        return image, pixel_size, oversampling

    def create_rgb_render(
        self, images, normalise=True, histogram_equalize=False, bit_depth=32
    ):
        def get_colors(num_colors):
            """Generate a list of colors for the given number of images."""
            colors = [
                (1.0, 0.0, 0.0),  # Red
                (0.0, 1.0, 0.0),  # Green
                (0.0, 0.0, 1.0),  # Blue
                (1.0, 1.0, 0.0),  # Yellow
                (1.0, 0.0, 1.0),  # Magenta
                (0.0, 1.0, 1.0),  # Cyan
            ]
            return colors[:num_colors]

        def normalise_image(image):
            """Normalize an image to the range [0, 1]."""
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val - min_val == 0:
                return image - min_val
            return (image - min_val) / (max_val - min_val)

        def histogram_equalization(image):
            """Apply histogram equalization to the image."""
            return cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0

        def to_8bit(image):
            """Convert floating-point image in range [0, 1] to 8-bit image."""
            return (image * 255).astype(np.uint8)

        def to_16bit(image):
            """Convert floating-point image in range [0, 1] to 16-bit image."""
            return (image * 65535).astype(np.uint16)

        def to_32bit(image):
            """Convert floating-point image in range [0, 1] to 32-bit image."""
            return (image * 4294967295).astype(np.uint32)

        try:
            # Determine the shape of the images
            Y, X = images[0].shape
            rgb = np.zeros((Y, X, 3), dtype=np.float32)
            colors = get_colors(len(images))

            for color, image in zip(colors, images):

                rgb[:, :, 0] += color[0] * image  # Red channel
                rgb[:, :, 1] += color[1] * image  # Green channel
                rgb[:, :, 2] += color[2] * image  # Blue channel

            rgb = np.clip(rgb, 0, 1)

            if bit_depth == 8:
                rgb = to_8bit(rgb)
            elif bit_depth == 8:
                rgb = to_16bit(rgb)
            elif bit_depth == 32:
                rgb = to_32bit(rgb)

        except Exception as e:
            print(traceback.format_exc())
            Y, X = images[0].shape
            if bit_depth == 8:
                rgb = np.zeros((Y, X, 3), dtype=np.uint8)
            elif bit_depth == 16:
                rgb = np.zeros((Y, X, 3), dtype=np.uint16)
            else:
                rgb = np.zeros((Y, X, 3), dtype=np.uint32)

        return rgb

    def picasso_render_finished(self):

        self.update_filter_criterion()
        self.update_criterion_ranges()

        self.update_ui(init=False)

    def draw_picasso_render(self, data):

        try:
            image, pixel_size, oversampling = data

            scale = [pixel_size / oversampling, pixel_size / oversampling]

            layer_names = [layer.name for layer in self.viewer.layers]

            if hasattr(self, "render_layer") == False:
                self.render_layer = self.viewer.add_image(
                    image,
                    name="SMLM Render",
                    colormap="inferno",
                    scale=scale,
                    blending="opaque",
                    rgb=True,
                )
                self.viewer.reset_view()
            else:

                if self.render_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.render_layer)

                self.render_layer = self.viewer.add_image(
                    image,
                    name="SMLM Render",
                    colormap="inferno",
                    scale=scale,
                    blending="opaque",
                    rgb=True,
                )
                self.viewer.reset_view()

            if self.gui.show_render.isChecked() == False:
                if self.render_layer in self.viewer.layers:
                    self.viewer.layers.remove(self.render_layer)

            self.render_layer.scale = scale
            self.render_layer.gamma = 0.2

        except:
            print(traceback.format_exc())

    def initialise_picasso_render(self):

        try:

            dataset = self.gui.picasso_render_dataset.currentText()
            channel = self.gui.picasso_render_channel.currentText()
            blur_method = self.gui.picasso_render_blur_method.currentText()
            min_blur_width = float(self.gui.picasso_render_min_blur.text())

            loc_data = self.get_locs(
                dataset, channel, return_dict=True, include_metadata=False
            )

            if len(loc_data) > 0:

                image_shape = loc_data[0]["image_shape"]
                if dataset != "All Datasets":
                    pixel_size = float(
                        self.dataset_dict[dataset]["pixel_size"]
                    )
                else:
                    dataset_name = list(self.dataset_dict.keys())[0]
                    pixel_size = float(
                        self.dataset_dict[dataset_name]["pixel_size"]
                    )

                if blur_method == "One-Pixel-Blur":
                    blur_method = "smooth"
                elif blur_method == "Global Localisation Precision":
                    blur_method = "convolve"
                elif blur_method == "Individual Localisation Precision, iso":
                    blur_method = "gaussian_iso"
                elif blur_method == "Individual Localisation Precision":
                    blur_method = "gaussian"
                else:
                    blur_method = None

                self.update_ui(init=True)

                worker = Worker(
                    self.render_picasso_locs,
                    loc_data=loc_data,
                    image_shape=image_shape,
                    blur_method=blur_method,
                    min_blur_width=min_blur_width,
                    pixel_size=pixel_size,
                )
                worker.signals.result.connect(self.draw_picasso_render)
                worker.signals.finished.connect(self.picasso_render_finished)
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()
