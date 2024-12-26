import pandas as pd
import glob
import os
from functools import reduce

def calculate_shared_SSMD(config):
    # List all CSV files starting with "SSMD"
    file_list = glob.glob(os.path.join(config.output_dir, "SSMD*.csv"))
    print(len(file_list))

    # Load and select only 'Probe.i' and 'Probe.j' columns from each file
    ssmd_probe_list = [pd.read_csv(file)[['Probe i', 'Probe j']] for file in file_list]

    # Concatenate all dataframes in the list
    combined_df = pd.concat(ssmd_probe_list, ignore_index=True)

    # Count occurrences of each (Probe i, Probe j) pair across the combined dataframe
    row_counts = combined_df.groupby(['Probe i', 'Probe j']).size().reset_index(name='count')

    # Calculate the XX% threshold based on the number of files
    num_files = len(file_list)
    print(num_files)
    print(config.precent_shared)
    threshold = config.precent_shared * num_files
    print(f'Count Rows That Repeat Over {threshold} Times')

    # Filter rows that appear in at least XX% of the files
    rows_in_threshold = row_counts[row_counts['count'] >= threshold]

    #return rows_in_threshold
    
    rows_in_threshold.to_csv(
        os.path.join(os.path.join(os.getcwd(), "shared_ssmd_data.csv")), index=False)
    print(f"Filtered probes saved as shared_ssmd_data, number of diff probes {len(rows_in_threshold)}")
