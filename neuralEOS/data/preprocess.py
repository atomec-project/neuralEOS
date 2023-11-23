import os
import fileinput
import time
import sys
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from atoMEC import Atom
from atoMEC.unitconv import K_to_ha, ev_to_ha, ha_to_gpa

K_to_ev = K_to_ha / ev_to_ha


class Preprocess:
    def __init__(self, parameters):
        self.parameters = parameters

    def cv_split(self):
        df_bins = self.make_pressure_bins()

        self.split_outer_nested(df_bins)

        return

    def make_pressure_bins(self):
        r"""Add an extra column which bins the pressures by their magnitude."""

        main_df = pd.read_csv(self.parameters.aa_pressure_file)

        main_df["P_class"] = 0
        bin_size = len(main_df) // self.parameters.num_bins
        main_df = main_df.sort_values(by=self.parameters.bin_partition)
        for i in range(self.parameters.num_bins - 1):
            main_df.loc[:, "P_class"].iloc[i * bin_size : (i + 1) * bin_size] = i + 1
        main_df.loc[:, "P_class"].iloc[(i + 1) * bin_size :] = self.parameters.num_bins

        main_df.to_csv(self.parameters.pressure_bins_file, index=False)
        return main_df

    def split_outer_nested(self, df_bins):
        r"""Make the outer split for the nested CV procedure."""

        skf = StratifiedKFold(
            n_splits=self.parameters.n_cv_outer,
            shuffle=True,
            random_state=self.parameters.random_state,
        )
        X = df_bins
        Y = df_bins.P_class

        for i, [train_index, test_index] in enumerate(skf.split(X, Y), 1):
            train_df = df_bins.iloc[train_index]
            test_df = df_bins.iloc[test_index]

            train_df.to_csv(
                self.parameters.aa_pressure_file[:-4] + "_train_" + str(i) + ".csv"
            )
            test_df.to_csv(
                self.parameters.aa_pressure_file[:-4] + "_test_" + str(i) + ".csv"
            )

        return
