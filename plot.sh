#!/bin/bash -e

source helper.sh && helper_setup

print_help() {
    echo "Usage: ${0} [OPTIONS]"
    echo "Options:"
    echo "  --result-dir   RESULT_DIR   relative path to directory having the result of the run.sh    (required)"
    echo "  --plot-dir     PLOT_DIR     relative path to directory having the result of the run.sh    (default replaces 'results' in the RESULT_DIR with 'plots')"
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --result-dir)
            result_dir="$2"
            shift 2
            ;;
        --plot-dir)
            plot_dir="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
    esac
done

if [[ -z ${result_dir} || ! -d ${result_dir} ]]; then
    echo "Improper argument: --result_dir ${result_dir}"
    echo "Either path not found or empty path"
    print_help
    exit 1
fi

if [[ -z ${plot_dir} ]]; then
    plot_dir=$(echo "${result_dir}" | sed 's|results|plots|')
fi

if [[ ${USE_DOCKER} -eq 1 ]]; then
    setup_tie_breaker_container
fi

for csv_path in $(ls ${result_dir}/*.csv)
do
    basename_csv_path=$(basename ${csv_path})
    basename_plot_path=$(echo "${basename_csv_path}" | sed 's|csv|png|')
    png_path=${plot_dir}/${basename_plot_path}

    if [[ ${USE_DOCKER} -eq 1 ]]; then
        docker_prefix="${DOCKER} exec -it ${TIE_BREAKER_CTR}"
    fi

    cmd="${docker_prefix} python3 src/plot.py \
        --csv-file-path ${csv_path} \
        --png-file-path ${png_path}"
    echo ${cmd}
    eval ${cmd}
done