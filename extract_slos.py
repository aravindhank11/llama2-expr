import pandas as pd

base_dir = "results/a100/"
models = ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'inception_v3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
batches=[2, 4, 8, 16, 32]

for model in models:
    for batch in batches:
        dir = base_dir + f"/{model}/{model}-{batch}-poisson/"
        total_p50_path = dir + "total_p50.csv"
        total_p90_path = dir + "total_p90.csv"

        df_p50 = pd.read_csv(total_p50_path)
        df_p90 = pd.read_csv(total_p90_path)

        print(f'{model}-{batch}: [{df_p50.iloc[:, -1].iloc[-1]}, {df_p90.iloc[:, -1].iloc[-1]}]')
