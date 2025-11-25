"""Script to stub out all remaining broken plot functions."""

import sys
sys.path.insert(0, "/Users/fdrgsp/Documents/git/cali/src")

# Check which files have the old dict[str, ROIData] signature that need updating
files_to_check = [
    "_plot_inferred_spike_synchrony.py",
    "_plot_inferred_spike_correlation.py", 
    "_plot_calcium_network_connectivity.py",
    "_plolt_evoked_experiment_data_plots.py",
]

for file in files_to_check:
    print(f"\n{file}:")
    try:
        with open(f"/Users/fdrgsp/Documents/git/cali/src/cali/plot/_single_wells_plots/{file}") as f:
            content = f.read()
            # Find all function definitions
            import re
            funcs = re.findall(r'def (_\w+)\([^)]*data: dict\[str, ROIData\]', content)
            for func in funcs:
                print(f"  - {func}")
    except FileNotFoundError:
        print(f"  File not found")
