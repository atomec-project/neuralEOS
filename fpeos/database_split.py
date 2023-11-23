# Define the lines that will be replaced
lines_to_replace = [
    "H_EOS_09-18-20.txt      H  1",
    "He_EOS_09-18-20.txt     He 1",
    "B_EOS_09-18-20.txt      B  1",
    "C_EOS_09-18-20.txt      C  1",
    "N_EOS_09-18-20.txt      N  1",
    "O_EOS_09-21-20.txt      O  1",
    "Ne_EOS_09-18-20.txt     Ne 1",
    "Na_EOS_09-18-20.txt     Na 1",
    "Mg_EOS_09-18-20.txt     Mg 1",
    "Al_EOS_09-18-20.txt     Al 1",
    "Si_EOS_10-19-20.txt     Si 1",
]

# Loop through the lines and create 5 new files
for i in range(5):
    new_lines = []
    for line in lines_to_replace:
        parts = line.split()
        if "09-18-20" in line:
            new_line = (
                f"{parts[0].replace('09-18-20', f'train_{i}')}    {parts[1]} {parts[2]}"
            )
        elif "09-21-20" in line:
            new_line = (
                f"{parts[0].replace('09-21-20', f'train_{i}')}    {parts[1]} {parts[2]}"
            )
        elif "10-19-20" in line:
            new_line = (
                f"{parts[0].replace('10-19-20', f'train_{i}')}    {parts[1]} {parts[2]}"
            )
        new_lines.append(new_line)

    # Write the new lines to a file
    with open(f"fpeos_database_{i}.txt", "w") as file:
        file.write("\n".join(new_lines))
