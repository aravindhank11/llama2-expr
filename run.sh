multiply_and_round() {
    local result=$(echo "$1 * $2" | bc)
    printf "%.2f\n" "$result"
}

device=v100
distri=poisson
modes=("mps-uncap" "orion" "ts")
models=("vision-mobilenet_v2-2-${distri}" "vision-resnet101-32-${distri}")
rps=(118.67 19.47)

start=0.1
end=1.5
step=0.1

rm print_outs.txt
# Loop through the values
for mode in ${modes[@]}
do
    for ((i = 1; i <= 15; i++)); do
        ratio=$(bc <<< "$start + ($i - 1) * $step")
        params=""
        for ((j=0; j<${#models[@]}; j++))
        do
            mul=$(multiply_and_round ${ratio} ${rps[${j}]})
            params="${params} ${models[${j}]}-${mul}"
        done
        cmd="./run_expr.sh ${device} ${mode} ${ratio} ${params}"
        echo "${mode} ${ratio}"
        eval ${cmd} >> print_outs.txt 2>&1
    done
done
