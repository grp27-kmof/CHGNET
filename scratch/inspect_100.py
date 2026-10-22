import os
import glob
import pandas as pd
from pymatgen.core import Structure

cif_files = glob.glob("INPUTS/cifs_100/*.cif")
data = []

for file in cif_files:
    basename = os.path.basename(file)
    mof_name = basename.replace(".cif", "")
    try:
        struct = Structure.from_file(file)
        volume = struct.volume
        density = struct.density
        num_atoms = len(struct)
        
        # Determine metal centers
        elements = [el.symbol for el in struct.composition.elements]
        metals = [el for el in elements if el in ["Fe", "Mn", "Ni", "Co", "Cu", "V", "Cr", "Zn"]]
        
        # Get topology name from mof_name, e.g. 0000[Co][dia]3[ASR]1 -> dia
        topology = "unknown"
        if "[" in mof_name:
            parts = mof_name.split("]")
            if len(parts) > 1 and "[" in parts[1]:
                topology = parts[1].split("[")[1]
        
        data.append({
            "mof": mof_name,
            "metals": ", ".join(metals),
            "volume": volume,
            "density": density,
            "num_atoms": num_atoms,
            "topology": topology
        })
    except Exception as e:
        print(f"Error parsing {file}: {e}")

df = pd.DataFrame(data)
df.to_csv("outputs/cifs_100_summary.csv", index=False)
print(f"Summarized {len(df)} CIF files. Top 15 by density (lowest first):")
print(df.sort_values(by="density").head(15))
