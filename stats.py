import pickle
import numpy as np
import sys
import os


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_stats(array):
    # Calculate percentiles
    min_val = np.min(array) * 1000
    percentile_50 = np.percentile(array, 50) * 1000
    percentile_90 = np.percentile(array, 90) * 1000
    percentile_99 = np.percentile(array, 99) * 1000
    max_val = np.max(array) * 1000

    # return results as string
    return "{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
        min_val, percentile_50, percentile_90, percentile_99, max_val
    )


def main():
    # Check if arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <mode> <pickle_file>")
        print(len(sys.argv))
        sys.exit(1)

    # Get file paths from command line arguments
    mode = sys.argv[1]
    pickle_file = sys.argv[2]

    # Validate file paths
    if not os.path.isfile(pickle_file):
        print(f"File '{pickle_file}' does not exist.")
        sys.exit(1)

    # Load arrays from pickle files
    model, tput, total_times, queued_times = load_pickle_file(pickle_file)
    total_time_stats = get_stats(total_times)
    queued_time_stats = get_stats(queued_times)
    print(
        "mode, model, tput, " +
        "lat_p0, lat_p50, lat_p90, lat_p99, lat_p100, " +
        "q_p0, q_p50, q_p90, q_p99, q_p100"
    )
    print("{}, {}, {:.2f}, {} {}".format(
        mode, model, tput, total_time_stats, queued_time_stats
    ))


if __name__ == "__main__":
    main()
