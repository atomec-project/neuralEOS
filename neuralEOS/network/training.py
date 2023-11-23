import numpy as np
import os.path
import random
import glob
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.model_selection import (
    RepeatedKFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import optuna
import pickle as pkl
import time
import sys
import string
import shutil

from neuralEOS import parameters as base_params


class Training:
    def __init__(self, parameters):
        self.parameters = parameters
        self.X_scaler = None
        self.Y_scaler = None
        self.X_scaled = None
        self.Y_scaled = None
        self.Y_scaled_pred = None
        self.output = None

    def submit_inner_training_jobs(self, training_file_prefix, savedir_prefix, use_aa):
        # serialize the parameters for command line arg to training script
        params_serial = pkl.dumps(self.parameters).hex()

        for i_cv in range(1, self.parameters.n_cv_outer + 1):
            cvpath = savedir_prefix + str(i_cv)
            training_file = training_file_prefix + str(i_cv) + ".csv"
            Path(savedir_prefix + str(i_cv)).mkdir(parents=True, exist_ok=True)
            slurm_train_ext = self.parameters.slurm_train_file.split("/")[-1]
            shutil.copyfile(
                self.parameters.slurm_train_file, cvpath + "/" + slurm_train_ext
            )
            os.chdir(cvpath)
            for _ in range(self.parameters.n_feature_trials):
                os_cmd = " ".join(
                    [
                        "sbatch",
                        slurm_train_ext,
                        params_serial,
                        cvpath,
                        training_file,
                        str(use_aa),
                    ]
                )
                os.popen(os_cmd)

    def submit_outer_training_jobs(
        self, training_file_prefix, savedir_prefix, model_file_prefix, use_aa
    ):
        # serialize the parameters for command line arg to training script
        params_serial = pkl.dumps(self.parameters).hex()

        for i_cv in range(1, self.parameters.n_cv_outer + 1):
            cvpath = savedir_prefix + str(i_cv) + "/"
            training_file = training_file_prefix + str(i_cv) + ".csv"
            slurm_train_ext = self.parameters.slurm_outer_train_file.split("/")[-1]
            os.chdir(self.parameters.scripts_dir)
            for i_ens in range(self.parameters.n_ensemble):
                model_file = model_file_prefix + str(i_cv) + "_" + str(i_ens) + ".pkl"
                os_cmd = " ".join(
                    [
                        "sbatch",
                        slurm_train_ext,
                        params_serial,
                        cvpath,
                        training_file,
                        model_file,
                        str(i_ens),
                        str(use_aa),
                    ]
                )
                os.popen(os_cmd)

    def submit_final_training_hyperopt_jobs(self, training_file, savedir, use_aa):
        # serialize the parameters for command line arg to training script
        params_serial = pkl.dumps(self.parameters).hex()

        Path(savedir).mkdir(parents=True, exist_ok=True)
        slurm_train_ext = self.parameters.slurm_train_file.split("/")[-1]
        shutil.copyfile(
            self.parameters.slurm_train_file, savedir + "/" + slurm_train_ext
        )
        os.chdir(savedir)
        for _ in range(self.parameters.n_feature_trials):
            os_cmd = " ".join(
                [
                    "sbatch",
                    slurm_train_ext,
                    params_serial,
                    savedir,
                    training_file,
                    str(use_aa),
                ]
            )
            os.popen(os_cmd)

    def submit_final_outer_training_jobs(
        self, training_file, savedir, model_file_prefix, use_aa
    ):
        # serialize the parameters for command line arg to training script
        params_serial = pkl.dumps(self.parameters).hex()

        slurm_train_ext = self.parameters.slurm_outer_train_file.split("/")[-1]
        os.chdir(self.parameters.scripts_dir)
        for i_ens in range(self.parameters.n_ensemble):
            model_file = model_file_prefix + str(i_ens) + ".pkl"
            os_cmd = " ".join(
                [
                    "sbatch",
                    slurm_train_ext,
                    params_serial,
                    savedir,
                    training_file,
                    model_file,
                    str(i_ens),
                    str(use_aa),
                ]
            )
            os.popen(os_cmd)

    def train_inner_loop(self, training_file, use_aa, save_dir=None):
        if use_aa:
            feature_list = self.parameters.feature_list_aa
            log_feature_map = self.parameters.log_feature_map_aa
        else:
            feature_list = self.parameters.feature_list_no_aa
            log_feature_map = self.parameters.log_feature_map_no_aa

        # use the training file name if no save folder provider
        if save_dir is None:
            save_dir = (
                self.parameters.cv_inner_path
                + "/"
                + training_file.split("/")[-1][:-4]
                + "/"
            )

        # load and sample (if required) the training df
        train_df = pd.read_csv(training_file)
        if self.parameters.n_train_samples is not None:
            train_df = train_df.sample(n=self.parameters.n_train_samples)

        # set up the inner CV object
        inner_cv = RepeatedStratifiedKFold(
            n_splits=self.parameters.n_cv_inner,
            n_repeats=self.parameters.n_cv_inner_repeats,
        )

        # set up lists / arrays for each random feature subset
        optimal_params = []
        features_all = []
        scores = np.zeros((1, 5))

        # randomly choose # of features
        N_features = list(
            np.random.choice(
                range(
                    self.parameters.n_features_min, self.parameters.n_features_max + 1
                ),
                1,
            )
        )[0]
        scores[:, 0] = N_features
        base_params.nn_input_length = N_features

        # select N_features randomly
        features = list(np.random.choice(feature_list, N_features, replace=False))
        print("final features", features)

        # select and scale the features
        (
            self.X_scaled,
            self.Y_scaled,
            self.X_scaler,
            self.Y_scaler,
        ) = self.select_scale_features(train_df, features, log_feature_map)

        # set up the model
        model = KerasRegressor(model=self.build_rnn, verbose=0)

        if self.parameters.error_metric == "MAPE":
            err_method = self.MAPE_scaled
        if self.parameters.error_metric == "SMAPE":
            err_method = self.SMAPE_scaled
        elif self.parameters.error_metric == "MAE":
            err_method = self.MAE_scaled
        elif self.parameters.error_metric == "adjMAE":
            err_method = self.MAE_scaled_adj

        # set up the optuna CV object
        clf = optuna.integration.OptunaSearchCV(
            estimator=model,
            param_distributions=self.parameters.network_param_distributions,
            cv=inner_cv.split(train_df, train_df.P_class),
            refit=False,
            return_train_score=False,
            n_jobs=1,
            verbose=1,
            n_trials=self.parameters.n_hyperopt_trials,
            scoring=make_scorer(err_method, greater_is_better=False),
        )

        # fit the optimum parameters
        clf.fit(self.X_scaled, self.Y_scaled)

        # get the best parameters
        if self.parameters.param_priority == "simple":
            best_params = self.get_best_params(clf)
        elif self.parameters.param_priority == "best":
            best_params = clf.best_params_

        # train an estimator on the best parameters
        best_estimator = KerasRegressor(
            model=self.build_rnn(
                best_params["model__neurons"],
                best_params["model__num_hidden"],
                best_params["model__learning_rate"],
                best_params["model__frac_decrease"],
            ),
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            verbose=False,
        )

        # fit, predict and save the scores
        best_estimator.fit(self.X_scaled, self.Y_scaled)
        # best_estimator.scores = scores
        # best_estimator.features = features
        # best_estimator.best_params = best_params
        # best_estimator.X_scaler = self.X_scaler
        # best_estimator.Y_scaler = self.Y_scaler
        # best_estimator.log_feature_map = log_feature_map
        self.Y_scaled_pred = best_estimator.predict(self.X_scaled)
        self.save_scores_as_pkl(scores, features, best_params, save_dir)
        # self.save_estimator(best_estimator, save_dir)

    def train_outer_loop(
        self,
        training_file,
        inner_scores_file,
        use_aa,
        save_dir=None,
        model_file=None,
        score_index=0,
    ):
        # load in the data
        if use_aa:
            feature_list = self.parameters.feature_list_aa
            log_feature_map = self.parameters.log_feature_map_aa
        else:
            feature_list = self.parameters.feature_list_no_aa
            log_feature_map = self.parameters.log_feature_map_no_aa

        # use the training file name if no save folder provider
        if save_dir is None:
            save_dir = (
                self.parameters.cv_inner_path
                + "/"
                + training_file.split("/")[-1][:-4]
                + "/"
            )

        # load the training df
        train_df = pd.read_csv(training_file)
        if self.parameters.n_train_samples is not None:
            train_df = train_df.sample(n=self.parameters.n_train_samples)

        # load in the df with the scores
        scores_df = pd.read_csv(inner_scores_file, sep=", ")
        scores_df = scores_df.dropna()

        # find the file with the best score
        if self.parameters.error_metric == "MAPE":
            scores_df = scores_df.sort_values(by="MAPE")
        if self.parameters.error_metric == "SMAPE":
            scores_df = scores_df.sort_values(by="SMAPE")
        elif self.parameters.error_metric == "MAE":
            scores_df = scores_df.sort_values(by="MAE")
        elif self.parameters.error_metric == "adjMAE":
            scores_df = scores_df.sort_values(by="adjMAE")

        best_filecode = scores_df.iloc[score_index].filecode

        # open and load the file
        with open(save_dir + "run_" + best_filecode + ".pkl", "rb") as f:
            # best_estimator = pkl.load(f)
            inner_cv_data = pkl.load(f)

        # retrieve the best parameters and features
        best_features = inner_cv_data["features"]
        best_params = inner_cv_data["best_params"]
        base_params.nn_input_length = len(best_features)

        # best_features = best_estimator.features

        # select and scale the features
        (
            self.X_scaled,
            self.Y_scaled,
            self.X_scaler,
            self.Y_scaler,
        ) = self.select_scale_features(train_df, best_features, log_feature_map)

        # train an estimator on the best parameters
        best_estimator = KerasRegressor(
            model=self.build_rnn(
                best_params["model__neurons"],
                best_params["model__num_hidden"],
                best_params["model__learning_rate"],
                best_params["model__frac_decrease"],
            ),
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            verbose=False,
        )
        best_estimator.fit(self.X_scaled, self.Y_scaled)

        # add the scalers and features to the model
        best_estimator.X_scaler = self.X_scaler
        best_estimator.Y_scaler = self.Y_scaler
        best_estimator.features = best_features
        best_estimator.log_feature_map = log_feature_map

        # save the estimator (model)
        if model_file is None:
            model_file = save_dir + "model_" + best_filecode + ".pkl"
        with open(model_file, "wb") as f:
            pkl.dump(best_estimator, f, protocol=pkl.HIGHEST_PROTOCOL)

        return

    def select_scale_features(
        self, train_df, features, log_feature_map, X_scaler=None, Y_scaler=None
    ):
        r"""Select and scale the features and output."""

        X_init = train_df.copy()
        X_init = X_init[features]
        for i, feature in enumerate(features):
            if log_feature_map[feature] == 1:
                absfeature = np.where(
                    np.abs(train_df[feature]) > 1e-4, np.abs(train_df[feature]), 1e-4
                )
                X_init[feature] = np.log10(absfeature)

        if X_scaler == None:
            X_scaler = preprocessing.StandardScaler().fit(X_init)
        # scale the features
        if self.parameters.scale_features:
            X_scaled = X_scaler.transform(X_init)
        else:
            X_scaled = X_init.values

        # scale the output
        if self.parameters.log_target:
            Y_init = np.log10(train_df[["P_ref"]]).values
        else:
            Y_init = train_df[["P_ref"]].values
        if Y_scaler == None:
            Y_scaler = preprocessing.StandardScaler().fit(Y_init)
        if self.parameters.scale_target:
            Y_scaled = Y_scaler.transform(Y_init)[:, 0]
        else:
            Y_scaled = Y_init[:, 0]

        return X_scaled, Y_scaled, X_scaler, Y_scaler

    def MAPE_scaled(self, y_true, y_pred):
        r"""Returns MAPE, with logarithmic and (optionally) standard scaling."""

        if self.parameters.scale_target:
            if self.parameters.log_target:
                y_true = 10 ** self.Y_scaler.inverse_transform([y_true])
                y_pred = 10 ** self.Y_scaler.inverse_transform([y_pred])
            else:
                y_true = self.Y_scaler.inverse_transform([y_true])
                y_pred = self.Y_scaler.inverse_transform([y_pred])

        else:
            if self.parameters.log_target:
                y_true = 10**y_true
                y_pred = 10**y_pred

        array_errors = np.abs((y_true - y_pred) / (y_true + 1e-6))

        MAPE = np.average(array_errors)

        return MAPE

    def SMAPE_scaled(self, y_true, y_pred):
        r"""Returns symmetric MAPE, with logarithmic and (optionally) standard scaling."""

        if self.parameters.scale_target:
            if self.parameters.log_target:
                y_true = 10 ** self.Y_scaler.inverse_transform([y_true])
                y_pred = 10 ** self.Y_scaler.inverse_transform([y_pred])
            else:
                y_true = self.Y_scaler.inverse_transform([y_true])
                y_pred = self.Y_scaler.inverse_transform([y_pred])

        else:
            if self.parameters.log_target:
                y_true = 10**y_true
                y_pred = 10**y_pred

        array_errors = np.abs(
            (y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
        )

        MAPE = np.average(array_errors)

        return MAPE

    def MAE_scaled(self, y_true, y_pred):
        r"""Returns MAE, with logarithmic and (optionally) standard scaling."""

        if self.parameters.scale_target:
            if self.parameters.log_target:
                y_true = self.Y_scaler.inverse_transform([y_true])
                y_pred = self.Y_scaler.inverse_transform([y_pred])
            else:
                y_true = np.log10(np.abs(self.Y_scaler.inverse_transform([y_true])))
                y_pred = np.log10(np.abs(self.Y_scaler.inverse_transform([y_pred])))

        else:
            if not self.parameters.log_target:
                y_true = np.log10(np.abs(y_true))
                y_pred = np.log10(np.abs(y_pred))

        MAE = np.average(np.abs(y_true - y_pred))

        return MAE

    def MAE_scaled_adj(self, y_true, y_pred):
        r"""Returns adjusted MAE, with logarithmic and (optionally) standard scaling."""

        if self.parameters.scale_target:
            if self.parameters.log_target:
                y_true = self.Y_scaler.inverse_transform([y_true])
                y_pred = self.Y_scaler.inverse_transform([y_pred])
            else:
                y_true = np.log10(np.abs(self.Y_scaler.inverse_transform([y_true])))
                y_pred = np.log10(np.abs(self.Y_scaler.inverse_transform([y_pred])))

        else:
            if not self.parameters.log_target:
                y_true = np.log10(np.abs(y_true))
                y_pred = np.log10(np.abs(y_pred))

        # gives more weight to higher pressures
        MAE = np.average(y_true * np.abs(y_true - y_pred))

        return MAE

    def get_best_params(self, clf):
        # define the "best" params based on least complex
        mean_test_scores = np.zeros((self.parameters.n_hyperopt_trials))
        mean_test_stderrs = np.zeros((self.parameters.n_hyperopt_trials))
        model_complexities = np.zeros((self.parameters.n_hyperopt_trials))
        for j in range(self.parameters.n_hyperopt_trials):
            trial = clf.trials_[j]
            # learning_rate = trial.params["model__learning_rate"]
            epochs = trial.params["epochs"]
            num_hidden = trial.params["model__num_hidden"]
            neurons = trial.params["model__neurons"]
            frac_decrease = trial.params["model__frac_decrease"]
            model_complexities[j] = self.model_complexity(
                neurons, num_hidden, epochs, frac_decrease
            )
            mean_test_scores[j] = -100 * trial.user_attrs["mean_test_score"]
            mean_test_stderrs[j] = (
                100
                * trial.user_attrs["std_test_score"]
                / np.sqrt(
                    self.parameters.n_cv_inner * self.parameters.n_cv_inner_repeats
                )
            )

        best_trial_index = np.argmin(mean_test_scores)
        max_err = (
            mean_test_scores[best_trial_index] + mean_test_stderrs[best_trial_index]
        )
        valid_indices = np.where(mean_test_scores < max_err)[0]
        valid_trials = []
        k = 0
        for j in valid_indices:
            if k == 0:
                min_complexity = model_complexities[j]
            complexity = model_complexities[j]
            if complexity < min_complexity:
                min_complexity = complexity
                best_trial_index = j
            k += 1

        best_trial = clf.trials_[best_trial_index]
        best_params = best_trial.params

        return best_params

    def save_scores_as_pkl(self, scores, features, best_params, save_dir):
        scores[:, 1] = 100 * self.MAPE_scaled(self.Y_scaled, self.Y_scaled_pred)
        scores[:, 2] = 200 * self.SMAPE_scaled(self.Y_scaled, self.Y_scaled_pred)
        scores[:, 3] = self.MAE_scaled(self.Y_scaled, self.Y_scaled_pred)
        scores[:, 4] = self.MAE_scaled_adj(self.Y_scaled, self.Y_scaled_pred)

        self.output = {
            "features": features,
            "best_params": best_params,
            "scores": scores,
            "timestamp": datetime.now(),
        }

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.parameters.cv_pkl_file is None:
            fname_scores = save_dir + "run_"
            +"".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            +".pkl"
        else:
            fname_scores = self.parameters.cv_pkl_file

        with open(fname_scores, "wb") as f:
            pkl.dump(self.output, f, protocol=pkl.HIGHEST_PROTOCOL)

    def save_estimator(self, best_estimator, save_dir):
        best_estimator.scores[:, 1] = 100 * self.MAPE_scaled(
            self.Y_scaled, self.Y_scaled_pred
        )
        best_estimator.scores[:, 2] = 200 * self.SMAPE_scaled(
            self.Y_scaled, self.Y_scaled_pred
        )
        best_estimator.scores[:, 3] = self.MAE_scaled(self.Y_scaled, self.Y_scaled_pred)
        best_estimator.scores[:, 4] = self.MAE_scaled_adj(
            self.Y_scaled, self.Y_scaled_pred
        )

        # self.output = {
        #     "features": features,
        #     "best_params": best_params,
        #     "scores": scores,
        #     "timestamp": datetime.now(),
        # }

        best_estimator.timestamp = datetime.now()

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.parameters.cv_pkl_file is None:
            fname_scores = save_dir + "run_"
            +"".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            +".pkl"
        else:
            fname_scores = self.parameters.cv_pkl_file

        with open(fname_scores, "wb") as f:
            pkl.dump(best_estimator, f, protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def gather_cv_scores(search_dir, filename_out=None, dt_init=None, dt_final=None):
        file_list = glob.glob(search_dir + "run_*.pkl")

        if dt_init is not None:
            dt_init_str = dt_init
            dt_init = datetime.strptime(dt_init, "%d/%m/%Y %H:%M")
        if dt_final is not None:
            dt_final_str = dt_final
            dt_final = datetime.strptime(dt_final, "%d/%m/%Y %H:%M")

        if filename_out is None:
            filename_out = (
                search_dir
                + "cv_results_"
                + "".join(random.choices(string.ascii_uppercase + string.digits, k=3))
                + ".txt"
            )

        summary_file = open(filename_out, "w")
        summary_file.write(
            "filecode, n_features, MAPE, SMAPE, MAE, adjMAE, timestamp \n"
        )

        for cv_file in file_list:
            with open(cv_file, "rb") as f:
                output = pkl.load(f)
            keep_file = True
            if dt_init is not None:
                if output["timestamp"] < dt_init:
                    keep_file = False
            if dt_final is not None:
                if output["timestamp"] > dt_final:
                    keep_file = False
            if keep_file:
                filecode = cv_file.split("/")[-1][4:-4:]
                scores = output["scores"][0]
                summary_file.write(
                    ", ".join(
                        [
                            filecode,
                            str(int(scores[0])),
                            str(round(scores[1], 2)),
                            str(round(scores[2], 2)),
                            str(round(scores[3], 4)),
                            str(round(scores[4], 4)),
                            output["timestamp"].strftime("%d/%m/%Y %H:%M"),
                        ]
                    )
                    + "\n"
                )

        summary_file.write(
            "\n" + "File generated: " + datetime.now().strftime("%d/%m/%Y %H:%M") + "\n"
        )
        if dt_init is not None:
            summary_file.write("Initial date and time: " + dt_init_str + "\n")
        if dt_final is not None:
            summary_file.write("Final date and time: " + dt_final_str)
        summary_file.close()

    @staticmethod
    def model_complexity(neurons, num_hidden, epochs, frac_decrease):
        return neurons * num_hidden * epochs * (frac_decrease) ** num_hidden

    @staticmethod
    def build_rnn(neurons, num_hidden, learning_rate, frac_decrease):
        model = Sequential()

        for i in range(num_hidden):
            model.add(
                Dense(
                    neurons,
                    input_shape=(base_params.nn_input_length,),
                    activation="relu",
                )
            )
            neurons *= frac_decrease

        model.add(Dense(1))
        model.compile(
            loss="mean_absolute_error",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"],
        )

        return model
