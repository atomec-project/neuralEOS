from pathlib import Path
import shutil
import sys
import fileinput
import os
import random

import numpy as np
import pandas as pd
from atoMEC import Atom
from atoMEC.postprocess import pressure
from atoMEC.unitconv import ev_to_ha, K_to_ha, angstrom_to_bohr, ha_to_gpa
from atoMEC import config as atoMEC_config

atoMEC_config.suppress_warnings = True


class GenerateAA:
    def __init__(self, parameters, from_Hu=False):
        self.parameters = parameters

    def make_all_files(self, from_Hu=False):
        """Make the atoMEC input files and (maybe) run them.

        Parameters
        ----------
        from_Hu : bool, optional
            if the data is from the Be Hu dataset
        """
        if from_Hu:
            pressure_file = self.parameters.Be_pressure_file
        else:
            pressure_file = self.parameters.pressure_file
        with open(pressure_file, "r") as f:
            lines = f.readlines()
            
        template_file = self.parameters.atoMEC_conv_template
        submit_script = self.parameters.atoMEC_slurm_script

        lines = lines[1:]  # ignore first text line

        # use for small tests
        if self.parameters.n_aa_samples is not None:
            random.seed(self.parameters.random_seed)
            lines = random.sample(lines, self.parameters.n_aa_samples)

        for line in lines:
            parts = line.split(",")
            species = parts[1]
            rho = parts[2]
            temp = parts[3]

            # skip species not in the element list
            if species not in self.parameters.element_list:
                continue

            # create directory (if not exists)
            dir_name = (
                self.parameters.aa_dir
                + species
                + "/"
                + "rho_"
                + str(round(float(rho), 3))
                + "/T_"
                + str(round(float(temp), 3))
                + "/"
                + self.parameters.xc_functional
                + "/"
            )
            Path(dir_name).mkdir(parents=True, exist_ok=True)

            # copy the template
            scf_outfile = dir_name + species + "_conv.py"
            shutil.copyfile(template_file, scf_outfile)

            # replace the relevant lines in the template
            for line in fileinput.input(scf_outfile, inplace=True):
                line = line.replace("density = 1", "density = " + str(rho))
                line = line.replace("temperature = 1", "temperature = " + str(temp))
                line = line.replace("species = x", "species = '" + str(species) + "'")
                sys.stdout.write(line)

            if self.parameters.make_slurm_conv:
                slurm_outfile = dir_name + "submit_conv_new.slurm"

                shutil.copyfile(
                    submit_script, slurm_outfile
                )

                for line in fileinput.input(slurm_outfile, inplace=True):
                    line = line.replace(
                        "#SBATCH --job-name=x",
                        "#SBATCH --job-name=" + species + rho[:3] + temp[:3],
                    )
                    line = line.replace(
                        "python -u x.py > x.log",
                        "python -u " + species + "_conv.py > " + species + "_conv.log",
                    )
                    sys.stdout.write(line)

            if self.parameters.submit_aa_jobs:
                cwd = os.getcwd()
                os.chdir(dir_name)
                if not os.path.exists("output.pkl"):
                    os.popen("sbatch submit_conv_new.slurm")
                else:
                    print("output.pkl already exists. Skipping sbatch command.")
                os.chdir(cwd)

    def extract_raw_fpeos(self):
        """Extract the data from the raw FPEOS files."""

        data_df = pd.DataFrame()

        for species in self.parameters.element_list:
            print("Extracting data for " + species)
            if species == "O":
                date = "09-21-20"
            elif species == "Si":
                date = "10-19-20"
            else:
                date = "09-18-20"
            fpeos_file = (
                self.parameters.fpeos_data_path + species + "_EOS_" + date + ".txt"
            )

            with open(fpeos_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                if "rho[g/cc]" in line:
                    info = line.split()

                    if "#" in info:
                        info = info[1:]

                    rho = float(info[5])
                    temp = float(info[9])
                    P_ref = float(info[11])
                    E_free = float(info[14])
                    vol = float(info[7])

                    atom = Atom(
                        species, temp, density=rho, units_temp="K", write_info=False
                    )
                    P_ion = pressure.ions_ideal(atom)
                    K_to_eV = K_to_ha / ev_to_ha
                    temp_eV = temp * K_to_eV

                    data_dict = {
                        "species": species,
                        "rho": rho,
                        "temp": temp_eV,
                        "Z": atom.at_chrg,
                        "vol": vol,
                        "AMU": atom.at_mass,
                        "P_ref": P_ref,
                        "P_ion": P_ion * ha_to_gpa,
                        "E_free": E_free,
                    }

                    df_row = pd.DataFrame(data_dict, index=[0])
                    data_df = pd.concat([data_df, df_row], ignore_index=True)

        data_df.to_csv(self.parameters.pressure_file)


    def extract_raw_fp_Be(self):
        """Extract the data from the raw FP-Be file."""

        print("Extracting data for Be")

        data_df = pd.DataFrame()

        fp_Be_file = self.parameters.fp_Be_filename 

        with open(fp_Be_file, "r") as f:
            lines = f.readlines()

        for line in lines[3:]:

            parts = line.split()

            if parts[0] != "ZONE":
                # ignore extrapolated values
                if parts[3] != "0":
                    rho =  float(parts[0])
                    temp = float(parts[1])
                    P_ref = 100 * float(parts[2])

                    atom = Atom(
                        "Be", temp, density=rho, units_temp="K", write_info=False
                    )
                    P_ion = pressure.ions_ideal(atom)
                    K_to_eV = K_to_ha / ev_to_ha
                    temp_eV = temp * K_to_eV
                    E_free = float(parts[4])
                    radius = atom.radius / angstrom_to_bohr
                    vol = (4./3.) * np.pi * radius**3
                    

                    data_dict = {
                        "species": "Be",
                        "rho": rho,
                        "temp": temp_eV,
                        "Z": atom.at_chrg,
                        "vol": vol,
                        "AMU": atom.at_mass,
                        "P_ref": P_ref,
                        "P_ion": P_ion * ha_to_gpa,
                        "E_free": E_free,
                    }

                    df_row = pd.DataFrame(data_dict, index=[0])
                    data_df = pd.concat([data_df, df_row], ignore_index=True)

        data_df.to_csv(self.parameters.Be_pressure_file)
