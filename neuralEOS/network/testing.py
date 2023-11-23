import pandas as pd
import pickle as pkl
import tabulate
import numpy as np

from . import training


class Testing:
    def __init__(self, parameters):
        self.parameters = parameters

    def predict(self, input_file, model_files, output_file=None):
        r"""With a pre-built model, predict some values."""

        if output_file is None:
            output_file = input_file

        # load in the file to make predictions on
        df_predict = pd.read_csv(input_file)
        df_predict["P_pred"] = np.zeros_like(df_predict["P_ref"])

        # load in the model
        if not isinstance(model_files, list):
            model_files = [model_files]

        N_pred = len(model_files)
            
        for i, model_file in enumerate(model_files):
            with open(model_file, "rb") as f:
                model = pkl.load(f)

            # set up the training object
            trainer = training.Training(self.parameters)

            # scale the test data correctly
            X_scaled, Y_scaled, X_scaler, Y_scaler = trainer.select_scale_features(
                df_predict,
                model.features,
                model.log_feature_map,
                X_scaler=model.X_scaler,
                Y_scaler=model.Y_scaler,
            )

            # predict the test data
            Y_scaled_pred = model.predict(X_scaled)

            # unscale the predictions
            Y_pred = self.unscale_pred(Y_scaled_pred, Y_scaler)

            # add the prediction to the dataframe
            df_predict["P_pred"] += pd.Series(Y_pred[0]) / N_pred

        # save the dataframe
        df_predict.to_csv(output_file)

    def unscale_pred(self, Y_scaled_pred, Y_scaler):
        if self.parameters.scale_target:
            if self.parameters.log_target:
                y_unscaled = 10 ** Y_scaler.inverse_transform([Y_scaled_pred])
            else:
                y_unscaled = Y_scaler.inverse_transform([Y_scaled_pred])
        else:
            if self.parameters.log_target:
                y_unscaled = 10**y_scaled

        return y_unscaled

    def evaluate(self, filelist, P_comp, P_ion=None, P_ref="P_ref", temp_thresh=None):
        # turn the filelist into a list if it's just one file
        if not isinstance(filelist, list):
            filelist = [filelist]

        MAPE_array = np.zeros((len(filelist)) + 2)
        MALE_array = np.zeros_like(MAPE_array)
        SMAPE_array = np.zeros_like(MAPE_array)
        MALE_adj_array = np.zeros_like(MAPE_array)
        f5_array = np.zeros_like(MAPE_array)
        f20_array = np.zeros_like(MAPE_array)
        RowIDs = filelist.copy()

        for i, Pfile in enumerate(filelist):
            df = pd.read_csv(Pfile)
            if temp_thresh is not None:
                df = df[df.temp>temp_thresh]
            if P_ion is not None:
                df[P_comp] += df[P_ion]
            MAPE_array[i] = self.calc_MAPE(df[P_ref], df[P_comp])
            MALE_array[i] = self.calc_MALE(df[P_ref], df[P_comp])
            SMAPE_array[i] = self.calc_SMAPE(df[P_ref], df[P_comp])
            MALE_adj_array[i] = self.calc_MALE_adj(df[P_ref], df[P_comp])
            f5_array[i], f20_array[i] = self.calc_fscores(df[P_ref], df[P_comp])
            RowIDs[i] = Pfile.split("/")[-1]

        # net summary stats
        MAPE_array[-2] = np.average(MAPE_array[:-2])
        MALE_array[-2] = np.average(MALE_array[:-2])
        SMAPE_array[-2] = np.average(SMAPE_array[:-2])
        MALE_adj_array[-2] = np.average(MALE_adj_array[:-2])
        MAPE_array[-1] = np.std(MAPE_array[:-2])
        MALE_array[-1] = np.std(MALE_array[:-2])
        MALE_adj_array[-1] = np.std(MALE_adj_array[:-2])
        SMAPE_array[-1] = np.std(SMAPE_array[:-2])
        f20_array[-1] = np.std(f20_array[:-2])
        f5_array[-1] = np.std(f5_array[:-2])
        f20_array[-2] = np.average(f20_array[:-2])
        f5_array[-2] = np.average(f5_array[:-2])

        # tabluate the results
        headers = ["MAPE (%)", "SMAPE (%)", "MALE", "MALE adj", "f5 score", "f20 score"]
        RowIDs.extend(["Average", "s.d."])
        tbl_data = np.vstack(
            (MAPE_array, SMAPE_array, MALE_array, MALE_adj_array, f5_array, f20_array)
        ).transpose()
        output_tbl = tabulate.tabulate(
            tbl_data,
            headers,
            tablefmt="presto",
            showindex=RowIDs,
            floatfmt="8.4f",
            stralign="right",
        )

        return output_tbl

    @staticmethod
    def calc_MAPE(P_ref, P_comp):
        APE = 100 * np.abs(P_ref - P_comp) / np.abs(P_ref)
        return np.average(APE)

    @staticmethod
    def calc_fscores(P_ref, P_comp):
        APE = 100 * np.abs(P_ref - P_comp) / np.abs(P_ref)
        f5 = 100 * np.size(np.where(APE <= 5)) / np.size(APE)
        f20 = 100 * np.size(np.where(APE <= 20)) / np.size(APE)
        return f5, f20

    @staticmethod
    def calc_SMAPE(P_ref, P_comp):
        SAPE = 200 * np.abs(P_ref - P_comp) / (np.abs(P_ref) + np.abs(P_comp))
        return np.average(SAPE)

    @staticmethod
    def calc_MALE(P_ref, P_comp):
        ALE = np.abs(np.log10(np.abs(P_ref)) - np.log10(np.abs(P_comp)))
        return np.average(ALE)

    @staticmethod
    def calc_MALE_adj(P_ref, P_comp):
        adj_ALE = np.log10(np.abs(P_ref)) * np.abs(
            np.log10(np.abs(P_ref)) - np.log10(np.abs(P_comp))
        )
        return np.average(adj_ALE)
