### Installation

Note: The following script is tested only on ubuntu 22.04 on AWS and on lamda

1) Install base packages
```
./install.sh
```

2) Install pip packages
```
./pip-install.sh
```

### Running Experiements

* *Input format*:
    ```
    ./run_expr.sh <MODE> <MODEL-1>-<BATCHSIZE-1> <MODEL-2>-<BATCHSIZE-2>
    ```

* *Non MIG Image classification Examples*
    ```
    ./run_expr.sh ts vgg11-16
    ./run_expr.sh mps-uncap vgg11-16 vgg11-16
    ./run_expr.sh mps-equi vgg11-16 vgg11-16 densenet121-32
    ./run_expr.sh mps-miglike densenet121-32 mobilenet_v2-1
    ```

* *MIG Image classification Examples*
    ```
    # Enable MIG and reboot if needed
    sudo nvidia-smi -i 0 -mig ENABLED
    sudo reboot now

    # Run the experiment
    ./run_expr.sh mig vgg11-16 vgg11-16
    ```

* *Llama examples*:
    ```
    # Format ./run_expr.sh <MODE> <MODEL-1>-<OUTPUT_TOKEN_SIZE-1> <MODEL-2>-<OUTPUT_TOKEN_SIZE-2>
    ./run_expr.sh mps-uncap llama-500 llama-500
    ```
    NOTE: Llama can be run simultaneously with image classifiers
