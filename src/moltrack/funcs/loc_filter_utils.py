import traceback
import numpy as np

class _loc_filter_utils:


    def pixseq_filter_localisations(self, viewer=None):

        try:

            dataset = self.gui.picasso_filter_dataset.currentText()
            criterion = self.gui.filter_criterion.currentText()
            min_value = self.gui.filter_min.value()
            max_value = self.gui.filter_max.value()

            if dataset in self.localisation_dict.keys():

                loc_dict = self.localisation_dict[dataset]

                if "localisations" in loc_dict.keys():

                    locs = loc_dict["localisations"].copy()

                    if len(locs) > 0:

                        columns = list(locs.dtype.names)

                        if criterion in columns:

                            self.gui.filter_localisations.setEnabled(False)

                            n_locs = len(locs)

                            locs = locs[locs[criterion] > min_value]
                            locs = locs[locs[criterion] < max_value]

                            n_filtered = len(locs)

                            if n_filtered < n_locs:

                                n_removed = n_locs - n_filtered

                                loc_dict["localisations"] = locs

                                self.localisation_dict[dataset] = loc_dict
                                self.draw_localisations(update_vis=True)

            self.update_criterion_ranges()
            print(f"Filtered {n_removed} localisations.")

            self.gui.filter_localisations.setEnabled(True)

        except:
            self.gui.filter_localisations.setEnabled(True)
            print(traceback.format_exc())


    def update_filter_dataset(self, viewer=None):

        if self.gui.picasso_filter_type.currentText() == "Localisations":
            self.gui.picasso_filter_dataset.setEnabled(True)
            self.gui.picasso_filter_dataset.show()
            self.gui.picasso_filter_dataset_label.show()
        else:
            self.gui.picasso_filter_dataset.setEnabled(False)
            self.gui.picasso_filter_dataset.hide()
            self.gui.picasso_filter_dataset_label.hide()

        self.update_filter_criterion()
        self.update_criterion_ranges()

    def update_filter_criterion(self, viewer=None):


        try:

            columns = []

            dataset = self.gui.picasso_filter_dataset.currentText()
            selector = self.gui.filter_criterion

            if dataset in self.localisation_dict.keys():

                loc_dict = self.localisation_dict[dataset]

                if "localisations" in loc_dict.keys():

                    locs = loc_dict["localisations"].copy()

                    if len(locs) > 0:

                        columns = list(locs.dtype.names)

                        columns = [col for col in columns if col not in ["dataset","channel"]]

            selector.clear()

            if len(columns) > 0:
                selector.addItems(columns)

        except:
            print(traceback.format_exc())


    def update_criterion_ranges(self, viewer=None, plot=True):

        try:

            self.filter_graph_canvas.clear()

            dataset = self.gui.picasso_filter_dataset.currentText()
            criterion = self.gui.filter_criterion.currentText()

            if dataset in self.localisation_dict.keys():

                loc_dict = self.localisation_dict[dataset]

                if "localisations" in loc_dict.keys():

                    locs = loc_dict["localisations"].copy()

                    if len(locs) > 0:

                        columns = list(locs.dtype.names)

                        if criterion in columns:

                            values = locs[criterion]

                            if values.dtype in [np.float32, np.float64, np.int32, np.int64]:

                                if plot:
                                    self.plot_filter_graph(criterion, values)

                                min_value = np.min(values)
                                max_value = np.max(values)

                                self.gui.filter_min.setMinimum(min_value)
                                self.gui.filter_min.setMaximum(max_value)

                                self.gui.filter_max.setMinimum(min_value)
                                self.gui.filter_max.setMaximum(max_value)

                                self.gui.filter_min.setValue(min_value)
                                self.gui.filter_max.setValue(max_value)

        except:
            print(traceback.format_exc())

    def plot_filter_graph(self, criterion = "", values = None):

        try:
            self.filter_graph_canvas.clear()

            if values is not None:

                values = values[~np.isnan(values)]

                if len(values) > 0:
                    ax = self.filter_graph_canvas.addPlot()

                    # Create histogram
                    y, x = np.histogram(values, bins=100)

                    ax.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 75))
                    ax.setLabel('bottom', f"{criterion} values")
                    ax.setLabel('left', 'Frequency')

        except:
            print(traceback.format_exc())



