from atoMEC import Atom, models, config
import numpy as np
import pickle as pkl
import time
import sys
import itertools

import neuralEOS

params = neuralEOS.Parameters()
params.grid_type = "sqrt"

config.numcores = -1
config.suppress_warnings = True

density = 1
temperature = 1
species = x

# set up the atom
atom = Atom(species, temperature, density=density, units_temp="eV")

# set up the model
model = models.ISModel(atom, bc="bands", unbound="quantum")

converger = neuralEOS.ConvergeAA(params)
conv_dict, output_dict = converger.converge_pressure_nconv(
    atom, model, grid_params={"ngrid_min": 500, "ngrid_max": 20000, "s0": 1e-4}
)

print(output_dict)
print(conv_dict)

# save the output
with open("conv_params.pkl", "wb") as f:
    pkl.dump(conv_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

with open("output.pkl", "wb") as f:
    pkl.dump(output_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
