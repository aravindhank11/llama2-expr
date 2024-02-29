multiply_and_round() {
    local result=$(echo "$1 * $2" | bc)
    printf "%.2f\n" "$result"
}

device=v100
mode=mps-uncap
distri=poisson
models=("vision-resnet101-32-${distri}" "vision-resnet101-32-${distri}" "vision-mobilenet_v2-2-${distri}")
rps=(12 10.5 48)

start=0.1
end=1.5
step=0.1

# Loop through the values
for ((i = 1; i <= 15; i++)); do
    ratio=$(bc <<< "$start + ($i - 1) * $step")
    params=""
    for ((j=0; j<${#models[@]}; j++))
    do
        mul=$(multiply_and_round ${ratio} ${rps[${j}]})
        params="${params} ${models[${j}]}-${mul}"
    done
    echo ${ratio}
    "./run_expr.sh ${device} ${mode} ${params}"
done
