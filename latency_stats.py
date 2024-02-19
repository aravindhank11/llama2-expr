import pickle
import numpy as np
import sys
import os

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    # Check if arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <pickle_file1> <pickle_file2> ...")
        sys.exit(1)

    # Get file paths from command line arguments
    pickle_files = sys.argv[1:]

    # Validate file paths
    for file_path in pickle_files:
        if not os.path.isfile(file_path):
            print(f"File '{file_path}' does not exist.")
            sys.exit(1)

    # Load arrays from pickle files
    arrays = []
    for file_path in pickle_files:
        array = load_pickle_file(file_path)
        arrays.append(array)

    # Concatenate arrays
    big_array = np.concatenate(arrays)

    # Calculate percentiles
    min_val = np.min(big_array) * 1000
    percentile_50 = np.percentile(big_array, 50) * 1000
    percentile_90 = np.percentile(big_array, 90) * 1000
    percentile_99 = np.percentile(big_array, 99) * 1000
    max_val = np.max(big_array) * 1000

    # Print results
    print("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
        min_val, percentile_50, percentile_90, percentile_99, max_val
    ))

if __name__ == "__main__":
    main()
