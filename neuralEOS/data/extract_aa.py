import pickle as pkl
import pandas as pd
import numpy as np
import scipy.integrate as integrate
from atoMEC.unitconv import ha_to_gpa, ev_to_ha
from atoMEC import Atom, models, config
from atoMEC.postprocess import pressure
from neuralEOS.network import testing
import tabulate
import os
import re
import sys


class ExtractAA:
    def __init__(self, parameters):
        self.parameters = parameters

    def extract_aa_features(self, from_Hu=False):
        if from_Hu:
            pressure_file = self.parameters.Be_pressure_file
        else:
            pressure_file = self.parameters.pressure_file
        with open(pressure_file, "r") as f:
            lines = f.readlines()
        lines = lines[1:]  # ignore first text line

        data_df = pd.DataFrame()

        for line in lines:
            # extract the information from the line
            parts = line.split(",")
            species = parts[1]
            rho = parts[2]
            temp = parts[3]
            Z = parts[4]
            vol = parts[5]
            AMU = parts[6]
            P_ref = parts[7]
            P_ion = float(parts[8])
            E_free_ref = parts[9]

            # skip species not in the element list
            if species not in self.parameters.element_list:
                print(species)
                continue

            dir_name_final = (
                self.parameters.aa_dir
                + species
                + "/rho_"
                + str(round(float(rho), 3))
                + "/T_"
                + str(round(float(temp), 3))
                + "/"
                + self.parameters.xc_functional
                + "/"
            )

            conv_file = dir_name_final + "conv_params.pkl"
            output_file = dir_name_final + "output.pkl"

            # read in output files if they exist; skip if not
            try:
                with open(conv_file, "rb") as f_conv:
                    conv_output = pkl.load(f_conv)
                    # print("scf found")
                with open(output_file, "rb") as f_out:
                    pressure_output = pkl.load(f_out)
            except FileNotFoundError:
                print("No pkl files", dir_name_final + species)
                continue

            # extract pressures and convert to gpa
            P_fd = pressure_output["P_fd_A"]
            P_st_tr = pressure_output["P_st_tr"]
            P_st_rr = pressure_output["P_st_rr"]
            P_vir_corr = pressure_output["P_vir_corr_A"]
            P_vir_nocorr = pressure_output["P_vir_nocorr_A"]
            P_elec_ideal = pressure_output["P_id"]
            MIS = pressure_output["MIS"]
            n_R = pressure_output["n_R"]
            v_R = pressure_output["V_R"]
            E_free = pressure_output["E_free"]
            chem_pot = pressure_output["chem_pot"]
            dn_dR = pressure_output["dn_dR"]
            dv_dR = pressure_output["dV_dR"]

            (
                P_fd_i,
                P_st_tr_i,
                P_st_rr_i,
                P_vir_corr_i,
                P_vir_nocorr_i,
                P_elec_ideal_i,
            ) = (
                P_fd + P_ion,
                P_st_tr + P_ion,
                P_st_rr + P_ion,
                P_vir_corr + P_ion,
                P_vir_nocorr + P_ion,
                P_elec_ideal + P_ion,
            )

            data_dict = {
                "species": species,
                "P_ref": float(P_ref),
                "rho": float(rho),
                "temp": float(temp),
                "P_aa_fd": P_fd,
                "P_aa_st_tr": P_st_tr,
                "P_aa_st_rr": P_st_rr,
                "P_aa_vir_corr": P_vir_corr,
                "P_aa_vir_nocorr": P_vir_nocorr,
                "P_aa_ideal": P_elec_ideal,
                "P_aa_fd_i": P_fd_i,
                "P_aa_st_tr_i": P_st_tr_i,
                "P_aa_st_rr_i": P_st_rr_i,
                "P_aa_vir_corr_i": P_vir_corr_i,
                "P_aa_vir_nocorr_i": P_vir_nocorr_i,
                "P_aa_ideal_i": P_elec_ideal_i,
                "P_ion": P_ion,
                "MIS": MIS,
                "n_R": n_R,
                "v_R": v_R,
                "Z": int(Z),
                "dn_dR": dn_dR,
                "dv_dR": dv_dR,
                "E_free": E_free,
                "chem_pot": chem_pot,
                "vol": float(vol),
                "E_free_ref": float(E_free_ref),
                "rho_Z": float(rho) / float(Z),
            }

            df_row = pd.DataFrame(data_dict, index=[0])
            data_df = pd.concat([data_df, df_row], ignore_index=True)
        
        if from_Hu:
            savedir = "/".join(self.parameters.aa_Be_pressure_file.split("/")[:-1])
            os.makedirs(savedir, exist_ok=True)
            print(self.parameters.aa_Be_pressure_file)
            data_df.to_csv(self.parameters.aa_Be_pressure_file)
        else:
            savedir = "/".join(self.parameters.aa_pressure_file.split("/")[:-1])
            os.makedirs(savedir, exist_ok=True)
            data_df.to_csv(self.parameters.aa_pressure_file)

    def extract_timings(self, timings_file, from_Hu=False):
        if from_Hu:
            pressure_file = self.parameters.Be_pressure_file
        else:
            pressure_file = self.parameters.pressure_file
        with open(pressure_file, "r") as f:
            lines = f.readlines()
        lines = lines[1:]  # ignore first text line

        data_df = pd.DataFrame()

        for line in lines:
            # extract the information from the line
            parts = line.split(",")
            parts = line.split(",")
            species = parts[1]
            rho = parts[2]
            temp = parts[3]
            P_ref = parts[7]

            # skip species not in the element list
            if species not in self.parameters.element_list:
                continue

            dir_name_final = (
                self.parameters.aa_dir
                + species
                + "/rho_"
                + str(round(float(rho), 3))
                + "/T_"
                + str(round(float(temp), 3))
                + "/"
                + self.parameters.xc_functional
                + "/"
            )

            output_file = dir_name_final + "output.pkl"
            log_file = dir_name_final + species + "_conv.log"

            # read in output files if they exist; skip if not
            try:
                with open(output_file, "rb") as f_out:
                    pressure_output = pkl.load(f_out)
            except FileNotFoundError:
                print("No pkl files", dir_name_final + species)
                continue

            with open(log_file, "r") as f:
                lines = f.readlines()

            # get the timings information
            pattern = re.compile(r"func:'CalcEnergy' took: (\d+\.\d+) sec")

            # Extract the last 3 matching lines
            matching_lines = [line for line in lines if pattern.match(line)]
            last_three = matching_lines[-3:]
            scf_final = matching_lines[-1]

            # Extract the number of seconds from each of the last 3 matching lines and sum them
            total_seconds = sum(
                float(pattern.search(line).group(1)) for line in last_three
            )

            scf_seconds = float(pattern.search(scf_final).group(1))

            data_dict = {
                "species": species,
                "rho": float(rho),
                "temp": float(temp),
                "tot_time": float(total_seconds),
                "scf_time": float(scf_seconds),
            }

            df_row = pd.DataFrame(data_dict, index=[0])
            data_df = pd.concat([data_df, df_row], ignore_index=True)

        data_df.to_csv(timings_file)


    def extract_conv_data(self, conv_data_file, from_Hu=False):
        if from_Hu:
            pressure_file = self.parameters.Be_pressure_file
        else:
            pressure_file = self.parameters.pressure_file
        with open(pressure_file, "r") as f:
            lines = f.readlines()
        lines = lines[1:]  # ignore first text line

        data_df = pd.DataFrame()

        for line in lines:
            # extract the information from the line
            parts = line.split(",")
            species = parts[1]
            rho = parts[2]
            temp = parts[3]
            P_ref = parts[7]

            # skip species not in the element list
            if species not in self.parameters.element_list:
                continue

            dir_name_final = (
                self.parameters.aa_dir
                + species
                + "/rho_"
                + str(round(float(rho), 3))
                + "/T_"
                + str(round(float(temp), 3))
                + "/"
                + self.parameters.xc_functional
                + "/"
            )

            output_file = dir_name_final + "conv_params.pkl"

            # read in output files if they exist; skip if not
            try:
                with open(output_file, "rb") as f_out:
                    conv_output = pkl.load(f_out)
            except FileNotFoundError:
                print("No pkl files", dir_name_final + species)
                continue

            df_row = pd.DataFrame(conv_output, index=[0])
            data_df = pd.concat([data_df, df_row], ignore_index=True)

        data_df.to_csv(conv_data_file)
