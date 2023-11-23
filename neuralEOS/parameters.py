import os

import optuna

# this is easier kept outside of the parameters class...
nn_input_length = 1


class Parameters:
    def __init__(self):
        # self.main_path = "/".join(os.getcwd().split("/")[:-1])
        self.main_path = os.path.expanduser("~/neuralEOS/")
        self.fig_path = self.main_path + "figs/"
        self.data_path = self.main_path + "data/"
        self.raw_data_path = self.data_path + "raw/"
        self.fpeos_data_path = self.raw_data_path + "FPEOS/"
        self.fp_Be_filename = self.raw_data_path + "Hu_Be_data.dat"
        self.processed_data_path = self.data_path + "processed/"
        self.intermediate_data_path = self.data_path + "intermediate/"
        self.model_path = self.main_path + "models/"
        self.element_list = [
            "H",
            "He",
            "B",
            "C",
            "N",
            "O",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
        ]
        self.species_list = [
            "H",
            "He",
            "B",
            "C",
            "N",
            "O",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "B4C",
            "BN",
            "C16H24",
            "C18H18",
            "C20H10",
            "LiF",
            "MgO",
            "MgSiO3",
        ]
        self.big_data_path = self.intermediate_data_path
        self.pressure_path = self.processed_data_path 
        self.cv_inner_path = self.intermediate_data_path + "inner_cv_output/"
        self.aa_dir = self.intermediate_data_path + "atoMEC_inputs/"
        self.templates_dir = self.main_path + "/templates/"
        self.scripts_dir = self.main_path + "/scripts/"
        self.elements_only = True
        self.pressure_file = self.pressure_path + "pressure_elements.csv"
        self.Be_pressure_file = self.pressure_path + "Be_Hu.csv"
        self.aa_pressure_file = self.pressure_path + "pressure_aa.csv"
        self.aa_Be_pressure_file = self.pressure_path + "pressure_aa_Be.csv"
        self.atoMEC_conv_template = self.scripts_dir + "atoMEC_conv.py"
        self.atoMEC_slurm_script = self.scripts_dir + "submit_atoMEC.slurm"
        self.num_bins = 10
        self.bin_partition = "P_ref"
        self.n_cv_outer = 5
        self.n_cv_inner = 5
        self.n_cv_inner_repeats = 3
        self.n_hyperopt_trials = 30
        self.n_feature_trials = 20
        self.n_features_min = 3
        self.n_features_max = 7
        self.random_state = 0
        self.feature_list_no_aa = ["temp", "rho", "Z", "vol", "P_ion", "rho_Z"]
        self.log_feature_map_no_aa = {
            "temp": 1,
            "rho": 1,
            "Z": 1,
            "vol": 1,
            "P_ion": 1,
            "rho_Z": 1,
        }
        self.feature_list_aa = [
            "P_aa_st_tr",
            "P_aa_st_rr",
            "P_aa_ideal",
            "P_aa_vir_corr",
            "P_aa_vir_nocorr",
            "P_aa_fd",
            "dn_dR",
            "dv_dR",
            "n_R",
            "temp",
            "rho_Z",
            "rho",
            "vol",
            "MIS",
        ]
        self.log_feature_map_aa = {
            "P_aa_st_tr": 1,
            "P_aa_st_rr": 1,
            "P_aa_ideal": 1,
            "P_aa_vir_corr": 1,
            "P_aa_vir_nocorr": 1,
            "P_aa_fd": 1,
            "P_aa_fd_i": 1,
            "P_aa_st_tr_i": 1,
            "P_aa_st_rr_i": 1,
            "P_aa_ideal_i": 1,
            "P_aa_vir_corr_i": 1,
            "P_aa_vir_nocorr_i": 1,
            "P_ion": 1,
            "dn_dR": 1,
            "dv_dR": 1,
            "n_R": 1,
            "temp": 1,
            "rho_Z": 1,
            "vol": 1,
            "rho": 1,
            "MIS": 1,
        }
        self.log_target = True
        self.scale_features = True
        self.scale_target = True
        self.network_param_distributions = {
            "epochs": optuna.distributions.IntDistribution(1000, 2000),
            "batch_size": optuna.distributions.IntDistribution(20, 60),
            "model__neurons": optuna.distributions.IntDistribution(20, 80),
            "model__learning_rate": optuna.distributions.FloatDistribution(
                0.0001, 0.001
            ),
            "model__frac_decrease": optuna.distributions.FloatDistribution(0.5, 1.0),
            "model__num_hidden": optuna.distributions.IntDistribution(1, 2),
        }
        self.ncores = -1
        self.n_train_samples = None
        self.n_test_samples = None
        self.n_aa_samples = None

        self.slurm_train_file = self.scripts_dir + "train_hyper_aa.slurm"
        self.slurm_outer_train_file = self.scripts_dir + "train_outer_loop.slurm"
        self.make_slurm_scf = True
        self.run_scf_calcs = False
        self.make_slurm_conv = True
        self.run_conv_calcs = False
        self.submit_aa_jobs = False
        self.make_slurm_pressure = True
        self.run_pressure_calcs = False
        self.xc_functional = "lda"
        self.copy_scf_files = False
        self.sum_e_ion_pressures = True
        self.param_priority = "best"
        self.error_metric = "adjMAE"
        self.cv_pkl_file = None
        self.grid_type = "sqrt"
        self.random_seed = 29
        self.n_ensemble = 3

    # @property
    # def pressure_file(self):
    #     if not self.elements_only:
    #         self._pressure_file = self.pressure_path + "pressure_species.csv"
    #     return self._pressure_file
