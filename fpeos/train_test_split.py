import random
import sys
kBoltzmann = 8.617333262145e-5  # Boltzmann constant in eV/K
element = sys.argv[1]
split_num = sys.argv[2]

# Seed to ensure reproducibility
random.seed((int(split_num)+2)**2)

# Lists to hold lines
header_lines = []
informative_lines = []

# Read the original file
with open(element + "_EOS_09-18-20.txt", "r") as infile:
    # Flag to indicate when to start collecting informative lines
    collect_lines = False

    for line in infile:
        # Check when to start collecting informative lines
        if "f= "+element in line:
            collect_lines = True

        if collect_lines:
            # Collect informative lines
            if line.strip():  # Ensure the line is not empty
                informative_lines.append(line)
        else:
            # Collect header lines
            header_lines.append(line)

# Shuffle the informative lines randomly
random.shuffle(informative_lines)

# Calculate index for the split
split_idx = int(len(informative_lines) * 0.8)

# Separate lines into training and testing sets
train_lines = informative_lines[:split_idx]
test_lines = informative_lines[split_idx:]

# Write the training set to a new file
with open(element + "_EOS_train_"+split_num+".txt", "w") as train_file:
    # Write header first
    for line in header_lines:
        train_file.write(line)
    # Write informative lines
    for line in train_lines:
        train_file.write(line)

# Write the testing set to a new file
with open(element + "_EOS_test_"+split_num+".txt", "w") as test_file:
    # Write header first
    for line in header_lines:
        test_file.write(line)
    # Write informative lines
    for line in test_lines:
        test_file.write(line)


# Lists to hold density and temperature values
densities = []
temperatures = []
pressures = []

# Read the original file
with open(element + "_EOS_test_"+split_num+".txt", "r") as infile:
    for line in infile:
        if "f= "+element in line:
            # Extract the density and temperature from the line
            tokens = line.split()
            for i, token in enumerate(tokens):
                if token == "rho[g/cc]=":
                    density = float(tokens[i + 1])
                elif token == "T[K]=":
                    temperature = float(tokens[i + 1])
                elif token == "P[GPa]=":
                    pressure = float(tokens[i+1])
            
            # Convert the temperature from K to eV
            temperature_eV = temperature * kBoltzmann
            
            # Store the density and converted temperature
            densities.append(density)
            temperatures.append(temperature_eV)
            pressures.append(pressure)

# Write to the new file
with open(element + "_EOS_test_input_"+split_num+".txt", "w") as outfile:
    outfile.write("{:<15} {:<15} {:<15}\n".format("rho", "temp", "pressure"))
    for density, temperature, pressure in zip(densities, temperatures, pressures):
        outfile.write("{:<15.6f} {:<15.6f} {:<15.6f}\n".format(density, temperature, pressure))

