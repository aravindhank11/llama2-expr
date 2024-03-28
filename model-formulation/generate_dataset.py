import pandas as pd
import statistics
import os
import shutil
import ast

RESULTS_DIR = '../results/a100'
# RESULTS_DIR = '/home/ps35324/a100'
PROFILE_DB = '../profiler/data/a100/model_profiles.csv'

def aggregate_mps(job_size):

    count = 0
    data_points = []
    for outer_dir in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, outer_dir)):
            models = outer_dir.split('-')
            if len(models) == job_size:
                for sub_dir in os.listdir(os.path.join(RESULTS_DIR, outer_dir)):
                    if 'closed' in sub_dir:
                        if len([entry for entry in os.listdir(os.path.join(RESULTS_DIR, outer_dir, sub_dir)) if os.path.isfile(os.path.join(RESULTS_DIR, outer_dir, sub_dir, entry))]) == 16:

                            df_tput = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/tput.csv')
                            df_process_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p0.csv')
                            df_process_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p50.csv')
                            df_process_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p90.csv')
                            df_process_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p99.csv')
                            df_process_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p100.csv')
                            df_total_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p0.csv')
                            df_total_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p50.csv')
                            df_total_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p90.csv')
                            df_total_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p99.csv')
                            df_total_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p100.csv')
                            df_queued_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p0.csv')
                            df_queued_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p50.csv')
                            df_queued_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p90.csv')
                            df_queued_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p99.csv')
                            df_queued_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p100.csv')

                            data_point = []
                            # Get model and batch sizes of the job mix
                            job_mix = df_tput.columns.tolist()[2:]
                            for job in job_mix:
                                data_point.append(job.split('-')[0][2:])
                                data_point.append(int(job.split('-')[1]))

                            # Start expanding out data
                            if 'mps-uncap' in df_tput['mode'].tolist():
                                if len(df_tput) == 2:
                                    print(os.path.join(RESULTS_DIR, outer_dir, sub_dir))
                                data_point.extend(df_tput[df_tput['mode'] == 'mps-uncap'].iloc[0])
                                data_point.extend(df_process_p0[df_process_p0['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p50[df_process_p50['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p90[df_process_p90['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p99[df_process_p99['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p100[df_process_p100['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p0[df_total_p0['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p50[df_total_p50['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p90[df_total_p90['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p99[df_total_p99['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p100[df_total_p100['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p0[df_queued_p0['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p50[df_queued_p50['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p90[df_queued_p90['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p99[df_queued_p99['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p100[df_queued_p100['mode'] == 'mps-uncap'].iloc[0,-int(f'{job_size}'):])
                                data_points.append(data_point)
                        else:
                            print(os.path.join(RESULTS_DIR, outer_dir, sub_dir))
    

    columns = []
    for i in range(1, job_size + 1):
        columns.append(f'm{i}')
        columns.append(f'b{i}')
    columns.append('mode')
    columns.append('load')
    metrics = ['tput', 'process_p0', 'process_p50', 'process_p90', 'process_p99', 'process_p100', 'total_p0', 'total_p50', 'total_p90', 'total_p99', 'total_p100', 'queued_p0', 'queued_p50', 'queued_p90', 'queued_p99', 'queued_p100']
    for metric in metrics:
        for i in range(1, job_size + 1):
            columns.append(f'{metric}_m{i}')

    df = pd.DataFrame(data_points, columns=columns)
    return df

def aggregate_mig():

    data_points = []
    for outer_dir in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, outer_dir)):
            models = outer_dir.split('-')
            if len(models) == 2:
                for sub_dir in os.listdir(os.path.join(RESULTS_DIR, outer_dir)):
                    if 'closed' in sub_dir:
                        if len([entry for entry in os.listdir(os.path.join(RESULTS_DIR, outer_dir, sub_dir)) if os.path.isfile(os.path.join(RESULTS_DIR, outer_dir, sub_dir, entry))]) == 16:
                            df_tput = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/tput.csv')
                            df_process_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p0.csv')
                            df_process_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p50.csv')
                            df_process_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p90.csv')
                            df_process_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p99.csv')
                            df_process_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p100.csv')
                            df_total_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p0.csv')
                            df_total_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p50.csv')
                            df_total_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p90.csv')
                            df_total_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p99.csv')
                            df_total_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p100.csv')
                            df_queued_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p0.csv')
                            df_queued_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p50.csv')
                            df_queued_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p90.csv')
                            df_queued_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p99.csv')
                            df_queued_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p100.csv')
                            dp1 = []
                            dp2 = []
                            for mode in df_tput['mode'].unique():
                                # Get model and batch sizes of the job mix
                                job_mix = df_tput.columns.tolist()[2:]
                                for i, job in enumerate(job_mix):
                                    if i == 0:
                                        dp1.append(job.split('-')[0][2:])
                                        dp1.append(int(job.split('-')[1]))
                                        dp1.append(4)
                                    elif i == 1:
                                        dp2.append(job.split('-')[0][2:])
                                        dp2.append(int(job.split('-')[1]))
                                        dp2.append(3)
                                
                                dp1.extend(df_tput[df_tput['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_tput[df_tput['mode'] == mode].iloc[0, 3:])

                                dp1.extend(df_total_p0[df_total_p0['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p0['mode'] == mode].iloc[0, 3:])
                                dp1.extend(df_total_p0[df_total_p50['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p50['mode'] == mode].iloc[0, 3:])
                                dp1.extend(df_total_p0[df_total_p90['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p90['mode'] == mode].iloc[0, 3:])
                                dp1.extend(df_total_p0[df_total_p99['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p99['mode'] == mode].iloc[0, 3:])
                                dp1.extend(df_total_p0[df_total_p100['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p100['mode'] == mode].iloc[0, 3:])

                                data_points.append(dp1)
                                data_points.append(dp2)

    count = 0
    for outer_dir in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, outer_dir)):
            models = outer_dir.split('-')
            if len(models) == 4:
                for sub_dir in os.listdir(os.path.join(RESULTS_DIR, outer_dir)):
                    if 'closed' in sub_dir:
                        if len([entry for entry in os.listdir(os.path.join(RESULTS_DIR, outer_dir, sub_dir)) if os.path.isfile(os.path.join(RESULTS_DIR, outer_dir, sub_dir, entry))]) == 16:
                            df_tput = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/tput.csv')
                            df_process_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p0.csv')
                            df_process_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p50.csv')
                            df_process_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p90.csv')
                            df_process_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p99.csv')
                            df_process_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/processing_p100.csv')
                            df_total_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p0.csv')
                            df_total_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p50.csv')
                            df_total_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p90.csv')
                            df_total_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p99.csv')
                            df_total_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/total_p100.csv')
                            df_queued_p0 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p0.csv')
                            df_queued_p50 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p50.csv')
                            df_queued_p90 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p90.csv')
                            df_queued_p99 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p99.csv')
                            df_queued_p100 = pd.read_csv(os.path.join(RESULTS_DIR, outer_dir, sub_dir) + '/queued_p100.csv')
                            dp1 = []
                            dp2 = []
                            for mode in df_tput['mode'].unique():
                                # Get model and batch sizes of the job mix
                                job_mix = df_tput.columns.tolist()[2:]
                                for i, job in enumerate(job_mix):
                                    if i == 0:
                                        dp1.append(job.split('-')[0][2:])
                                        dp1.append(int(job.split('-')[1]))
                                        dp1.append(2)
                                    elif i == 3:
                                        dp2.append(job.split('-')[0][2:])
                                        dp2.append(int(job.split('-')[1]))
                                        dp2.append(1)
                                
                                dp1.extend(df_tput[df_tput['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_tput[df_tput['mode'] == mode].iloc[0, 5:])

                                dp1.extend(df_total_p0[df_total_p0['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p0['mode'] == mode].iloc[0, 5:])
                                dp1.extend(df_total_p0[df_total_p50['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p50['mode'] == mode].iloc[0, 5:])
                                dp1.extend(df_total_p0[df_total_p90['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p90['mode'] == mode].iloc[0, 5:])
                                dp1.extend(df_total_p0[df_total_p99['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p99['mode'] == mode].iloc[0, 5:])
                                dp1.extend(df_total_p0[df_total_p100['mode'] == mode].iloc[0, 2:3])
                                dp2.extend(df_total_p0[df_total_p100['mode'] == mode].iloc[0, 5:])

                                data_points.append(dp1)
                                data_points.append(dp2)

    for data in data_points:
        if len(data) == 18:
            print(data)

    columns = ['model', 'bs', 'mig_slice', 'tput', 'total_p0', 'total_p50', 'total_p90', 'total_p99', 'total_p100']
    df = pd.DataFrame(data_points, columns=columns)

    return df

def aggregate_mig_2():
    base_dir = "../results/a100/"
    models = ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'inception_v3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
    batches=[2, 4, 8, 16, 32]
    count = 0
    data_points = []
    # Get slice data for 4 and 3
    for model in models:
        for batch in batches:
            dir = base_dir + f"/{model}-{model}/{model}-{batch}-closed_{model}-{batch}-closed"
            df_tput = pd.read_csv(dir + '/tput.csv')
            df_process_p0 = pd.read_csv(dir + '/processing_p0.csv')
            df_process_p50 = pd.read_csv(dir + '/processing_p50.csv')
            df_process_p90 = pd.read_csv(dir + '/processing_p90.csv')
            df_process_p99 = pd.read_csv(dir + '/processing_p99.csv')
            df_process_p100 = pd.read_csv(dir + '/processing_p100.csv')
            df_total_p0 = pd.read_csv(dir + '/total_p0.csv')
            df_total_p50 = pd.read_csv(dir + '/total_p50.csv')
            df_total_p90 = pd.read_csv(dir + '/total_p90.csv')
            df_total_p99 = pd.read_csv(dir + '/total_p99.csv')
            df_total_p100 = pd.read_csv(dir + '/total_p100.csv')
            df_queued_p0 = pd.read_csv(dir + '/queued_p0.csv')
            df_queued_p50 = pd.read_csv(dir + '/queued_p50.csv')
            df_queued_p90 = pd.read_csv(dir + '/queued_p90.csv')
            df_queued_p99 = pd.read_csv(dir + '/queued_p99.csv')
            df_queued_p100 = pd.read_csv(dir + '/queued_p100.csv')
            dp1 = []
            dp2 = []

            # Get model and batch sizes of the job mix
            job_mix = df_tput.columns.tolist()[2:]
            for i, job in enumerate(job_mix):
                if i == 0:
                    dp1.append(job.split('-')[0][2:])
                    dp1.append(int(job.split('-')[1]))
                    dp1.append(4)
                elif i == 1:
                    dp2.append(job.split('-')[0][2:])
                    dp2.append(int(job.split('-')[1]))
                    dp2.append(3)
            
            dp1.extend(df_tput[df_tput['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_tput[df_tput['mode'] == 'mig'].iloc[0, 3:])

            dp1.extend(df_total_p0[df_total_p0['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p0['mode'] == 'mig'].iloc[0, 3:])
            dp1.extend(df_total_p0[df_total_p50['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p50['mode'] == 'mig'].iloc[0, 3:])
            dp1.extend(df_total_p0[df_total_p90['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p90['mode'] == 'mig'].iloc[0, 3:])
            dp1.extend(df_total_p0[df_total_p99['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p99['mode'] == 'mig'].iloc[0, 3:])
            dp1.extend(df_total_p0[df_total_p100['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p100['mode'] == 'mig'].iloc[0, 3:])

            data_points.append(dp1)
            data_points.append(dp2)
    
    for model in models:
        for batch in batches:
            dir = base_dir + f"/{model}-{model}-{model}-{model}/{model}-{batch}-closed_{model}-{batch}-closed_{model}-{batch}-closed_{model}-{batch}-closed"
            df_tput = pd.read_csv(dir + '/tput.csv')
            df_process_p0 = pd.read_csv(dir + '/processing_p0.csv')
            df_process_p50 = pd.read_csv(dir + '/processing_p50.csv')
            df_process_p90 = pd.read_csv(dir + '/processing_p90.csv')
            df_process_p99 = pd.read_csv(dir + '/processing_p99.csv')
            df_process_p100 = pd.read_csv(dir + '/processing_p100.csv')
            df_total_p0 = pd.read_csv(dir + '/total_p0.csv')
            df_total_p50 = pd.read_csv(dir + '/total_p50.csv')
            df_total_p90 = pd.read_csv(dir + '/total_p90.csv')
            df_total_p99 = pd.read_csv(dir + '/total_p99.csv')
            df_total_p100 = pd.read_csv(dir + '/total_p100.csv')
            df_queued_p0 = pd.read_csv(dir + '/queued_p0.csv')
            df_queued_p50 = pd.read_csv(dir + '/queued_p50.csv')
            df_queued_p90 = pd.read_csv(dir + '/queued_p90.csv')
            df_queued_p99 = pd.read_csv(dir + '/queued_p99.csv')
            df_queued_p100 = pd.read_csv(dir + '/queued_p100.csv')
            dp1 = []
            dp2 = []

            # Get model and batch sizes of the job mix
            job_mix = df_tput.columns.tolist()[2:]
            for i, job in enumerate(job_mix):
                if i == 0:
                    dp1.append(job.split('-')[0][2:])
                    dp1.append(int(job.split('-')[1]))
                    dp1.append(2)
                elif i == 3:
                    dp2.append(job.split('-')[0][2:])
                    dp2.append(int(job.split('-')[1]))
                    dp2.append(1)
            
            dp1.extend(df_tput[df_tput['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_tput[df_tput['mode'] == 'mig'].iloc[0, 5:])

            dp1.extend(df_total_p0[df_total_p0['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p0['mode'] == 'mig'].iloc[0, 5:])
            dp1.extend(df_total_p0[df_total_p50['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p50['mode'] == 'mig'].iloc[0, 5:])
            dp1.extend(df_total_p0[df_total_p90['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p90['mode'] == 'mig'].iloc[0, 5:])
            dp1.extend(df_total_p0[df_total_p99['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p99['mode'] == 'mig'].iloc[0, 5:])
            dp1.extend(df_total_p0[df_total_p100['mode'] == 'mig'].iloc[0, 2:3])
            dp2.extend(df_total_p0[df_total_p100['mode'] == 'mig'].iloc[0, 5:])
            
            if len(dp1) != 9:
                if count == 0:
                    print(dp1)
                    print(dir)
            if len(dp2) !=9:
                if count == 0:
                    print(dp2)
                    print(dir)
            count += 1

            data_points.append(dp1)
            data_points.append(dp2)
    # for data in data_points:
    #     if len(data) != 9:
    #         print(data)
    columns = ['model', 'bs', 'mig_slice', 'tput', 'total_p0', 'total_p50', 'total_p90', 'total_p99', 'total_p100']
    df = pd.DataFrame(data_points, columns=columns)

    return df

def lookup_slo(row, job_no):
    MODEL_CONFIGS_250 = [['mobilenet_v2', 32, 23450.0], ['densenet121', 32, 59535.0], ['densenet201', 32, 94150.0], ['vgg19', 32, 77805.0], ['inception_v3', 32, 39270.0], ['resnet50', 32, 46585.0], ['resnet18', 32, 17955.0], ['resnet34', 32, 25375.0], ['resnet101', 32, 70035.0], ['resnet152', 32, 95725.0], ['densenet161', 32, 111125.0], ['densenet169', 32, 75495.0], ['vgg11', 32, 39620.0], ['vgg13', 32, 57400.0], ['vgg16', 32, 67620.0], ['alexnet', 32, 10920.0], ['squeezenet1_0', 32, 22785.0], ['squeezenet1_1', 32, 15995.0], ['mobilenet_v3_large', 32, 25025.0], ['mobilenet_v3_small', 32, 20230.0], ['efficientnet_b5', 32, 106645.0], ['efficientnet_b6', 32, 140455.0], ['efficientnet_b7', 32, 187285.0], ['mobilenet_v2', 2, 20650.0], ['densenet121', 2, 55545.0], ['densenet201', 2, 95025.0], ['vgg19', 2, 8855.0], ['inception_v3', 2, 40180.0], ['resnet50', 2, 24115.0], ['resnet18', 2, 8330.0], ['resnet34', 2, 13825.0], ['resnet101', 2, 43785.0], ['resnet152', 2, 63560.0], ['densenet161', 2, 78120.0], ['densenet169', 2, 78890.0], ['vgg11', 2, 5530.0], ['vgg13', 2, 6755.0], ['vgg16', 2, 7770.0], ['alexnet', 2, 3570.0], ['squeezenet1_0', 2, 9485.0], ['squeezenet1_1', 2, 8680.0], ['mobilenet_v3_large', 2, 24395.0], ['mobilenet_v3_small', 2, 20895.0], ['efficientnet_b5', 2, 74445.0], ['efficientnet_b6', 2, 85575.0], ['efficientnet_b7', 2, 102865.0], ['mobilenet_v2', 4, 20370.0], ['densenet121', 4, 55545.0], ['densenet201', 4, 95200.0], ['vgg19', 4, 13020.0], ['inception_v3', 4, 40425.0], ['resnet50', 4, 23625.0], ['resnet18', 4, 8295.0], ['resnet34', 4, 13720.0], ['resnet101', 4, 43785.0], ['resnet152', 4, 64085.0], ['densenet161', 4, 77210.0], ['densenet169', 4, 78855.0], ['vgg11', 4, 7560.0], ['vgg13', 4, 10010.0], ['vgg16', 4, 11515.0], ['alexnet', 4, 3570.0], ['squeezenet1_0', 4, 9590.0], ['squeezenet1_1', 4, 8820.0], ['mobilenet_v3_large', 4, 24605.0], ['mobilenet_v3_small', 4, 20755.0], ['efficientnet_b5', 4, 73149.99999999999], ['efficientnet_b6', 4, 85855.0], ['efficientnet_b7', 4, 107660.0], ['mobilenet_v2', 8, 20475.0], ['densenet121', 8, 55685.0], ['densenet201', 8, 95760.0], ['vgg19', 8, 23275.0], ['inception_v3', 8, 40880.0], ['resnet50', 8, 23975.0], ['resnet18', 8, 8960.0], ['resnet34', 8, 14035.0], ['resnet101', 8, 44135.0], ['resnet152', 8, 64155.0], ['densenet161', 8, 77105.0], ['densenet169', 8, 78680.0], ['vgg11', 8, 13055.0], ['vgg13', 8, 17815.0], ['vgg16', 8, 20475.0], ['alexnet', 8, 4585.0], ['squeezenet1_0', 8, 9765.0], ['squeezenet1_1', 8, 8855.0], ['mobilenet_v3_large', 8, 24710.0], ['mobilenet_v3_small', 8, 21385.0], ['efficientnet_b5', 8, 74760.0], ['efficientnet_b6', 8, 87815.0], ['efficientnet_b7', 8, 107310.0], ['mobilenet_v2', 16, 20440.0], ['densenet121', 16, 56980.0], ['densenet201', 16, 93695.0], ['vgg19', 16, 41370.0], ['inception_v3', 16, 40355.0], ['resnet50', 16, 26390.0], ['resnet18', 16, 10535.0], ['resnet34', 16, 15085.0], ['resnet101', 16, 43995.0], ['resnet152', 16, 64330.0], ['densenet161', 16, 76965.0], ['densenet169', 16, 78470.0], ['vgg11', 16, 21770.0], ['vgg13', 16, 30835.0], ['vgg16', 16, 36050.0], ['alexnet', 16, 6755.0], ['squeezenet1_0', 16, 13020.0], ['squeezenet1_1', 16, 9660.0], ['mobilenet_v3_large', 16, 25585.0], ['mobilenet_v3_small', 16, 21490.0], ['efficientnet_b5', 16, 78610.0], ['efficientnet_b6', 16, 90055.0], ['efficientnet_b7', 16, 110390.0]]
    new_slos = {'mobilenet_v2-2': [11.06250286102295, 28.213596343994148], 'mobilenet_v2-4': [7.328152656555176, 20.46399116516113], 'mobilenet_v2-8': [9.105205535888672, 28.820157051086422], 'mobilenet_v2-16': [10.136961936950684, 25.838327407836918], 'mobilenet_v2-32': [21.3320255279541, 89.20621871948254], 'mobilenet_v3_small-2': [9.493708610534668, 30.624675750732408], 'mobilenet_v3_small-4': [6.962418556213379, 18.46108436584473], 'mobilenet_v3_small-8': [10.103464126586914, 31.44035339355469], 'mobilenet_v3_small-16': [7.981538772583008, 24.18406009674073], 'mobilenet_v3_small-32': [8.991241455078125, 24.157381057739254], 'mobilenet_v3_large-2': [10.19287109375, 25.970196723937992], 'mobilenet_v3_large-4': [10.463953018188477, 31.406164169311523], 'mobilenet_v3_large-8': [8.74340534210205, 21.95785045623779], 'mobilenet_v3_large-16': [9.174346923828123, 23.64635467529297], 'mobilenet_v3_large-32': [18.35155487060547, 53.41248512268066], 'densenet121-2': [31.86607360839844, 82.40795135498047], 'densenet121-4': [40.970563888549805, 130.66999912261963], 'densenet121-8': [20.061135292053223, 61.494946479797385], 'densenet121-16': [34.93523597717285, 91.96534156799316], 'densenet121-32': [55.29773235321045, 161.38911247253418], 'densenet161-2': [31.385421752929688, 97.31435775756836], 'densenet161-4': [34.93690490722656, 104.54678535461426], 'densenet161-8': [39.22939300537109, 130.18898963928223], 'densenet161-16': [81.49385452270508, 211.2916231155396], 'densenet161-32': [150.1929759979248, 402.6620864868164], 'densenet169-2': [32.2953462600708, 93.69852542877194], 'densenet169-4': [37.70208358764648, 111.1121892929078], 'densenet169-8': [37.24122047424317, 112.37168312072754], 'densenet169-16': [72.13079929351807, 169.94497776031494], 'densenet169-32': [77.0866870880127, 235.37182807922363], 'densenet201-2': [55.79066276550293, 137.5442028045655], 'densenet201-4': [42.35696792602539, 113.26956748962402], 'densenet201-8': [46.94116115570069, 133.82303714752197], 'densenet201-16': [62.50441074371338, 216.6438102722168], 'densenet201-32': [140.98715782165527, 507.3609590530398], 'vgg11-2': [4.491686820983887, 12.486314773559576], 'vgg11-4': [4.9076080322265625, 14.774084091186523], 'vgg11-8': [10.160207748413086, 29.346108436584476], 'vgg11-16': [20.45154571533203, 65.66710472106938], 'vgg11-32': [40.741920471191406, 130.5216789245609], 'vgg13-2': [5.352497100830078, 14.85733985900879], 'vgg13-4': [7.332563400268555, 24.69758987426758], 'vgg13-8': [20.986437797546387, 61.68134212493898], 'vgg13-16': [25.80809593200684, 71.98476791381837], 'vgg13-32': [57.21712112426758, 176.62100791931155], 'vgg16-2': [6.448507308959961, 18.3983325958252], 'vgg16-4': [9.12332534790039, 28.58603000640869], 'vgg16-8': [18.642187118530277, 53.190302848815925], 'vgg16-16': [36.31937503814697, 116.04452133178712], 'vgg16-32': [59.967875480651855, 190.44158458709745], 'vgg19-2': [6.924629211425781, 17.524337768554688], 'vgg19-4': [11.4821195602417, 31.886911392211918], 'vgg19-8': [25.303125381469727, 86.88945770263673], 'vgg19-16': [47.20258712768555, 161.4960193634034], 'vgg19-32': [69.72730159759521, 184.2193603515625], 'inception_v3-2': [15.114545822143556, 40.23642539978028], 'inception_v3-4': [18.36538314819336, 55.87575435638427], 'inception_v3-8': [18.77737045288086, 52.09059715271001], 'inception_v3-16': [24.31881427764893, 91.33017063140876], 'inception_v3-32': [39.029598236083984, 125.16593933105467], 'resnet18-2': [3.32498550415039, 8.381843566894531], 'resnet18-4': [4.45866584777832, 12.677240371704103], 'resnet18-8': [5.157947540283203, 15.062761306762695], 'resnet18-16': [7.94684886932373, 23.18594455718994], 'resnet18-32': [16.067981719970703, 41.36037826538086], 'resnet34-2': [6.753325462341309, 21.199584007263184], 'resnet34-4': [6.506919860839844, 19.17881965637207], 'resnet34-8': [8.031487464904785, 22.634100914001465], 'resnet34-16': [12.609004974365234, 35.66241264343262], 'resnet34-32': [23.201465606689453, 60.73904037475586], 'resnet50-2': [15.022516250610352, 47.57630825042726], 'resnet50-4': [15.934944152832031, 48.004198074340856], 'resnet50-8': [31.15856647491455, 100.51560401916504], 'resnet50-16': [33.52701663970947, 110.16175746917725], 'resnet50-32': [71.20847702026367, 551.6562938690189], 'resnet101-2': [37.6279354095459, 98.98934364318852], 'resnet101-4': [32.98497200012207, 103.6813735961914], 'resnet101-8': [41.705965995788574, 134.9987268447876], 'resnet101-16': [48.97916316986084, 134.917950630188], 'resnet101-32': [100.55983066558838, 559.5760583877563], 'resnet152-2': [53.725600242614746, 128.05020809173587], 'resnet152-4': [52.88875102996826, 147.14336395263672], 'resnet152-8': [59.587717056274414, 178.08079719543457], 'resnet152-16': [65.30427932739258, 180.2811622619629], 'resnet152-32': [130.81252574920654, 467.5929307937622], 'alexnet-2': [2.0967721939086914, 5.826306343078614], 'alexnet-4': [15.523195266723633, 111.34619712829594], 'alexnet-8': [618.685245513916, 1469.8224782943723], 'alexnet-16': [9.903192520141602, 800.369453430176], 'alexnet-32': [11.010289192199709, 27.551174163818366], 'squeezenet1_0-2': [5.701303482055664, 15.452766418457037], 'squeezenet1_0-4': [6.941795349121094, 17.28706359863282], 'squeezenet1_0-8': [8.197307586669922, 23.445844650268555], 'squeezenet1_0-16': [21.50583267211914, 115.78102111816403], 'squeezenet1_0-32': [61.8593692779541, 328.75738143920904], 'squeezenet1_1-2': [7.618069648742676, 19.27053928375244], 'squeezenet1_1-4': [5.745649337768555, 19.358348846435547], 'squeezenet1_1-8': [7.340192794799805, 20.30348777770996], 'squeezenet1_1-16': [15.970110893249512, 680.0486087799072], 'squeezenet1_1-32': [38.78581523895264, 3191.081166267395], 'efficientnet_b5-2': [62.55030632019043, 227.14972496032715], 'efficientnet_b5-4': [52.23727226257324, 122.29514122009276], 'efficientnet_b5-8': [48.195481300354, 137.98916339874268], 'efficientnet_b5-16': [71.48623466491699, 190.3698444366455], 'efficientnet_b5-32': [88.68300914764404, 229.1651964187622], 'efficientnet_b6-2': [49.16810989379883, 121.783447265625], 'efficientnet_b6-4': [72.48008251190186, 192.4673318862915], 'efficientnet_b6-8': [64.6524429321289, 187.1220111846924], 'efficientnet_b6-16': [59.7454309463501, 131.84561729431152], 'efficientnet_b6-32': [353.773832321167, 1243.6289310455325], 'efficientnet_b7-2': [68.09377670288086, 199.70269203186052], 'efficientnet_b7-4': [73.36020469665527, 187.29534149169928], 'efficientnet_b7-8': [99.68793392181396, 230.86111545562756], 'efficientnet_b7-16': [131.41703605651855, 542.326378822327], 'efficientnet_b7-32': [250.2284049987793, 584.7785949707031], 'llama-2': [359.54952239990234, 363.2411956787109], 'llama-4': [711.1155986785889, 716.2215709686279], 'llama-8': [1432.9147338867188, 1437.9785060882568], 'llama-16': [2847.921848297119, 2856.339454650879], 'llama-32': [5645.272731781006, 5662.619113922119], 'bert-2': [38.394927978515625, 38.75255584716797], 'bert-4': [41.44906997680664, 41.65172576904297], 'bert-8': [59.00382995605469, 59.201717376708984], 'bert-16': [117.26856231689452, 117.48790740966797], 'bert-32': [200.26922225952148, 202.4393081665039]}
    slos = {}
    for item in MODEL_CONFIGS_250:
        slos[f'{item[0]}-{item[1]}'] = item[2]

    model_batch = str(row[f'm{job_no}']) + '-' + str(int(row[f'b{job_no}']))
    # return new_slos[model_batch] / 1000
    return new_slos[model_batch]

def compute_slo_violation(row, job_no):
    slo_violation = 0
    if row[f'slo_m{job_no}'] < row[f'total_p100_m{job_no}']:
        slo_violation = 1
    return slo_violation

def compute_slo_violation_new_slos(row, job_no):
    slo_violation = 0
    if (row[f'slo_m{job_no}'][1]) < row[f'total_p100_m{job_no}']:
        slo_violation = 1
    return slo_violation
    
def add_slos_to_data(df, job_size):
    df_slos = df.copy()
    for i in range(1, job_size + 1):
        df_slos[f'slo_m{i}'] = df_slos.apply(lambda row: lookup_slo(row, i), axis=1)
        df_slos[f'slo_violation_m{i}'] = df_slos.apply(lambda row: compute_slo_violation_new_slos(row, i), axis=1)


    slo_cols = [f'slo_violation_m{i}' for i in range(1, job_size + 1)]
    df_slos['slo_violation_no'] = df_slos[slo_cols].sum(axis=1)
    return df_slos

def stitch_mig_data(df, df_mig, job_size):

    m1s = []
    b1s = []
    m2s = []
    b2s = []
    # m3s = []
    # b3s = []
    modes = []
    loads = []
    tput_m1s = []
    tput_m2s = []
    # tput_m3s = []
    total_p90_m1s = []
    total_p90_m2s = []
    # total_p90_m3s = []
    total_p99_m1s = []
    total_p99_m2s = []
    # total_p99_m3s = []
    total_p100_m1s = []
    total_p100_m2s = []
    # total_p100_m3s = []
    slo_m1s = []
    slo_violation_m1s = []
    slo_m2s = []
    slo_violation_m2s = []
    # slo_m3s = []
    # slo_violation_m3s = []
    slo_violation_nos = []
    with open(f'./job_mixes/combos_{job_size}.txt', 'r') as file:
        found = 1
        count = 0
        for line in file:
            split_line = line.split(" ")
            m1 = split_line[0].split("-")[0]
            b1 = int(split_line[0].split("-")[1])
            slo_m1 = float(split_line[0].split("-")[2]) / 1000

            m2 = split_line[1].split("-")[0]
            b2 = int(split_line[1].split("-")[1])
            slo_m2 = float(split_line[1].split("-")[2]) / 1000

            # m3 = split_line[2].split("-")[0]
            # b3 = int(split_line[2].split("-")[1])
            # slo_m3 = float(split_line[2].split("-")[2]) / 1000
            
            # if len(df[(df['m1'] == m1) & (df['b1'] == b1) & (df['slo_m1'] == slo_m1) & (df['m2'] == m2) & (df['b2'] == b2) & (df['slo_m2'] == slo_m2) & (df['m3'] == m3) & (df['b3'] == b3) & (df['slo_m3'] == slo_m3)]) != 0:
            # if len(df[(df['m1'] == m1) & (df['b1'] == b1) & (df['slo_m1'] == slo_m1) & (df['m2'] == m2) & (df['b2'] == b2) & (df['slo_m2'] == slo_m2)]) != 0:
            if len(df[(df['m1'] == m1) & (df['b1'] == b1) & (df['m2'] == m2) & (df['b2'] == b2)]) != 0:                
                m1_slice = int(split_line[0].split("-")[3])
                m2_slice = int(split_line[1].split("-")[3])
                # m3_slice = int(split_line[2].split("-")[3])

                m1s.append(m1)
                b1s.append(b1)
                m2s.append(m2)
                b2s.append(b2)
                # m3s.append(m3)
                # b3s.append(b3)
                modes.append('mig')
                loads.append(1.0)
                tput_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['tput'].iloc[0])
                tput_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['tput'].iloc[0])
                # tput_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['tput'].iloc[0])
                total_p90_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p90'].iloc[0])
                total_p90_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p90'].iloc[0])
                # total_p90_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p90'].iloc[0])
                total_p99_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p99'].iloc[0])
                total_p99_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p99'].iloc[0])
                # total_p99_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p99'].iloc[0])
                total_p100_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p100'].iloc[0])
                total_p100_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p100'].iloc[0])
                # total_p100_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p100'].iloc[0])
                slo_m1s.append(-1)
                slo_m2s.append(-1)
                # slo_m3s.append(-1)
                slo_violation_m1s.append(-1)
                slo_violation_m2s.append(-1)
                # slo_violation_m3s.append(-1)
                slo_violation_nos.append(-1)
            else:
                break
    
    data = {'m1': m1s, 'b1': b1s, 'm2': m2s, 'b2': b2s, 'mode': modes, 'load': loads, 'tput_m1': tput_m1s, 'tput_m2': tput_m2s, 'total_p90_m1': total_p90_m1s, 'total_p90_m2': total_p90_m2s,
            'total_p99_m1': total_p99_m1s, 'total_p99_m2': total_p99_m2s, 'total_p100_m1': total_p100_m1s, 'total_p100_m2': total_p100_m2s, 'slo_m1': slo_m1s, 'slo_violation_m1': slo_violation_m1s,
            'slo_m2': slo_m2s, 'slo_violation_m2': slo_violation_m2s, 'slo_violation_no': slo_violation_nos}

    # data = {'m1': m1s, 'b1': b1s, 'm2': m2s, 'b2': b2s, 'm3': m3s, 'b3': b3s, 'mode': modes, 'load': loads, 'tput_m1': tput_m1s, 'tput_m2': tput_m2s, 'tput_m3': tput_m3s, 'total_p90_m1': total_p90_m1s, 'total_p90_m2': total_p90_m2s, 'total_p90_m3': total_p90_m3s,
    #         'total_p99_m1': total_p99_m1s, 'total_p99_m2': total_p99_m2s, 'total_p99_m3': total_p99_m3s, 'total_p100_m1': total_p100_m1s, 'total_p100_m2': total_p100_m2s, 'total_p100_m3': total_p100_m3s, 'slo_m1': slo_m1s, 'slo_violation_m1': slo_violation_m1s,
    #         'slo_m2': slo_m2s, 'slo_violation_m2': slo_violation_m2s, 'slo_m3': slo_m3s, 'slo_violation_m3': slo_violation_m3s, 'slo_violation_no': slo_violation_nos}

    df_mig_add = pd.DataFrame(data)
    df_final = pd.concat([df, df_mig_add])
    # df_final['job_mix'] = df_final.apply(lambda row: '-'.join([str(row['m1']), str(row['b1']), str(row['m2']), str(row['b2']), str(row['m3']), str(row['b3'])]), axis=1)
    df_final['job_mix'] = df_final.apply(lambda row: '-'.join([str(row['m1']), str(row['b1']), str(row['m2']), str(row['b2'])]), axis=1)

    return df_final

def merge_data():
    df1 = pd.read_csv('./data/raw-data/tb_250_data_new_slos.csv')
    df2 = pd.read_csv('./data/raw-data/tb_2400_data_new_slos.csv')

    df1['job_mix'] = df1.apply(lambda row: '-'.join([str(row['m1']), str(row['b1']), str(row['m2']), str(row['b2']), str(row['m3']), str(row['b3'])]), axis=1)
    df_final = pd.concat([df1, df2])
    df_final.to_csv('./data/raw-data/tb-large-training-set-3-new-slos.csv', index=False)

def label_data(job_size):

    df = pd.read_csv('./data/raw-data/tb_2400_data_size_2.csv')
    if job_size == 3:
        df = pd.read_csv('./data/raw-data/tb-large-training-set-3-new-slos.csv')
    labels = {}
    data_points = []

    for job_mix in df['job_mix'].unique():
        df_mps = df[(df['job_mix'] == job_mix) & (df['mode'] == 'mps-uncap')]
        df_mig = df[(df['job_mix'] == job_mix) & (df['mode'] == 'mig')]
        mps_tput = df_mps[[f"tput_m{i}" for i in range(1, job_size + 1)]].sum().sum()
        mps_tail_latency = df_mps[[f"total_p100_m{i}" for i in range(1, job_size + 1)]].max().max()
        mps_slo_violations = df_mps['slo_violation_no'].tolist()[0]
        mig_tput = df_mig[[f"tput_m{i}" for i in range(1, job_size + 1)]].sum().sum()
        mig_tail_latency = df_mig[[f"total_p100_m{i}" for i in range(1, job_size + 1)]].max().max()
        mig_slo_violations = df_mig['slo_violation_no'].tolist()[0]

        data_point = []
        # If one has an SLO violation, choose other
        if (mps_slo_violations > 0) and (mig_slo_violations == 0):
            data_point.extend(df_mig.iloc[0].tolist())
            data_point.append(0)
            data_point.append('mps_slo_violation')
        elif (mps_slo_violations == 0) and (mig_slo_violations > 0):
            data_point.extend(df_mps.iloc[0].tolist())
            data_point.append(1)
            data_point.append('mig_slo_violation')
        elif (mps_slo_violations == 0) and (mig_slo_violations == 0):
            if (mps_tput > mig_tput):
                data_point.extend(df_mps.iloc[0].tolist())
                data_point.append(1)
                data_point.append('mps_tput')
            else:
                data_point.extend(df_mig.iloc[0].tolist())
                data_point.append(0)
                data_point.append('mig_tput')
            # labels[job_mix] = (mps_tput > mig_tput).astype(int)
        elif (mps_slo_violations > 0) and (mig_slo_violations > 0):
            if (mps_tail_latency < mig_tail_latency):
                data_point.extend(df_mps.iloc[0].tolist())
                data_point.append(1)
                data_point.append('mps_tail')
            else:
                data_point.extend(df_mig.iloc[0].tolist())
                data_point.append(0)
                data_point.append('mig_tail')
        data_points.append(data_point)
            # labels[job_mix] = (mps_tail_latency < mig_tail_latency).astype(int)
    

    columns = df.columns.tolist() + ['label', 'case']
    df_final = pd.DataFrame(data_points, columns=columns)
    print(df_final['label'].value_counts().reset_index(name='count').to_string())
    print(df_final['case'].value_counts().reset_index(name='count').to_string())
    return df_final

def add_profiling_data(df_init, job_size):
    df_profiled = pd.read_csv(PROFILE_DB)
    df = df_init
    for i in range(1, job_size + 1):
        df = pd.merge(df, df_profiled, left_on=[f'm{i}', f'b{i}'], right_on=['model', 'batch_size'], how='left')
        new_column_names = {col: col + f'_{i}' for col in df_profiled.columns}
        df.rename(columns=new_column_names, inplace=True)
        df.drop([f'model_{i}', f'batch_size_{i}', f'tput_m{i}', f'total_p90_m{i}', f'total_p99_m{i}', f'total_p100_m{i}'], axis=1, inplace=True)
    return df

def get_new_slo(row):
    new_slos = {'mobilenet_v2-2': [11.06250286102295, 28.213596343994148], 'mobilenet_v2-4': [7.328152656555176, 20.46399116516113], 'mobilenet_v2-8': [9.105205535888672, 28.820157051086422], 'mobilenet_v2-16': [10.136961936950684, 25.838327407836918], 'mobilenet_v2-32': [21.3320255279541, 89.20621871948254], 'mobilenet_v3_small-2': [9.493708610534668, 30.624675750732408], 'mobilenet_v3_small-4': [6.962418556213379, 18.46108436584473], 'mobilenet_v3_small-8': [10.103464126586914, 31.44035339355469], 'mobilenet_v3_small-16': [7.981538772583008, 24.18406009674073], 'mobilenet_v3_small-32': [8.991241455078125, 24.157381057739254], 'mobilenet_v3_large-2': [10.19287109375, 25.970196723937992], 'mobilenet_v3_large-4': [10.463953018188477, 31.406164169311523], 'mobilenet_v3_large-8': [8.74340534210205, 21.95785045623779], 'mobilenet_v3_large-16': [9.174346923828123, 23.64635467529297], 'mobilenet_v3_large-32': [18.35155487060547, 53.41248512268066], 'densenet121-2': [31.86607360839844, 82.40795135498047], 'densenet121-4': [40.970563888549805, 130.66999912261963], 'densenet121-8': [20.061135292053223, 61.494946479797385], 'densenet121-16': [34.93523597717285, 91.96534156799316], 'densenet121-32': [55.29773235321045, 161.38911247253418], 'densenet161-2': [31.385421752929688, 97.31435775756836], 'densenet161-4': [34.93690490722656, 104.54678535461426], 'densenet161-8': [39.22939300537109, 130.18898963928223], 'densenet161-16': [81.49385452270508, 211.2916231155396], 'densenet161-32': [150.1929759979248, 402.6620864868164], 'densenet169-2': [32.2953462600708, 93.69852542877194], 'densenet169-4': [37.70208358764648, 111.1121892929078], 'densenet169-8': [37.24122047424317, 112.37168312072754], 'densenet169-16': [72.13079929351807, 169.94497776031494], 'densenet169-32': [77.0866870880127, 235.37182807922363], 'densenet201-2': [55.79066276550293, 137.5442028045655], 'densenet201-4': [42.35696792602539, 113.26956748962402], 'densenet201-8': [46.94116115570069, 133.82303714752197], 'densenet201-16': [62.50441074371338, 216.6438102722168], 'densenet201-32': [140.98715782165527, 507.3609590530398], 'vgg11-2': [4.491686820983887, 12.486314773559576], 'vgg11-4': [4.9076080322265625, 14.774084091186523], 'vgg11-8': [10.160207748413086, 29.346108436584476], 'vgg11-16': [20.45154571533203, 65.66710472106938], 'vgg11-32': [40.741920471191406, 130.5216789245609], 'vgg13-2': [5.352497100830078, 14.85733985900879], 'vgg13-4': [7.332563400268555, 24.69758987426758], 'vgg13-8': [20.986437797546387, 61.68134212493898], 'vgg13-16': [25.80809593200684, 71.98476791381837], 'vgg13-32': [57.21712112426758, 176.62100791931155], 'vgg16-2': [6.448507308959961, 18.3983325958252], 'vgg16-4': [9.12332534790039, 28.58603000640869], 'vgg16-8': [18.642187118530277, 53.190302848815925], 'vgg16-16': [36.31937503814697, 116.04452133178712], 'vgg16-32': [59.967875480651855, 190.44158458709745], 'vgg19-2': [6.924629211425781, 17.524337768554688], 'vgg19-4': [11.4821195602417, 31.886911392211918], 'vgg19-8': [25.303125381469727, 86.88945770263673], 'vgg19-16': [47.20258712768555, 161.4960193634034], 'vgg19-32': [69.72730159759521, 184.2193603515625], 'inception_v3-2': [15.114545822143556, 40.23642539978028], 'inception_v3-4': [18.36538314819336, 55.87575435638427], 'inception_v3-8': [18.77737045288086, 52.09059715271001], 'inception_v3-16': [24.31881427764893, 91.33017063140876], 'inception_v3-32': [39.029598236083984, 125.16593933105467], 'resnet18-2': [3.32498550415039, 8.381843566894531], 'resnet18-4': [4.45866584777832, 12.677240371704103], 'resnet18-8': [5.157947540283203, 15.062761306762695], 'resnet18-16': [7.94684886932373, 23.18594455718994], 'resnet18-32': [16.067981719970703, 41.36037826538086], 'resnet34-2': [6.753325462341309, 21.199584007263184], 'resnet34-4': [6.506919860839844, 19.17881965637207], 'resnet34-8': [8.031487464904785, 22.634100914001465], 'resnet34-16': [12.609004974365234, 35.66241264343262], 'resnet34-32': [23.201465606689453, 60.73904037475586], 'resnet50-2': [15.022516250610352, 47.57630825042726], 'resnet50-4': [15.934944152832031, 48.004198074340856], 'resnet50-8': [31.15856647491455, 100.51560401916504], 'resnet50-16': [33.52701663970947, 110.16175746917725], 'resnet50-32': [71.20847702026367, 551.6562938690189], 'resnet101-2': [37.6279354095459, 98.98934364318852], 'resnet101-4': [32.98497200012207, 103.6813735961914], 'resnet101-8': [41.705965995788574, 134.9987268447876], 'resnet101-16': [48.97916316986084, 134.917950630188], 'resnet101-32': [100.55983066558838, 559.5760583877563], 'resnet152-2': [53.725600242614746, 128.05020809173587], 'resnet152-4': [52.88875102996826, 147.14336395263672], 'resnet152-8': [59.587717056274414, 178.08079719543457], 'resnet152-16': [65.30427932739258, 180.2811622619629], 'resnet152-32': [130.81252574920654, 467.5929307937622], 'alexnet-2': [2.0967721939086914, 5.826306343078614], 'alexnet-4': [15.523195266723633, 111.34619712829594], 'alexnet-8': [618.685245513916, 1469.8224782943723], 'alexnet-16': [9.903192520141602, 800.369453430176], 'alexnet-32': [11.010289192199709, 27.551174163818366], 'squeezenet1_0-2': [5.701303482055664, 15.452766418457037], 'squeezenet1_0-4': [6.941795349121094, 17.28706359863282], 'squeezenet1_0-8': [8.197307586669922, 23.445844650268555], 'squeezenet1_0-16': [21.50583267211914, 115.78102111816403], 'squeezenet1_0-32': [61.8593692779541, 328.75738143920904], 'squeezenet1_1-2': [7.618069648742676, 19.27053928375244], 'squeezenet1_1-4': [5.745649337768555, 19.358348846435547], 'squeezenet1_1-8': [7.340192794799805, 20.30348777770996], 'squeezenet1_1-16': [15.970110893249512, 680.0486087799072], 'squeezenet1_1-32': [38.78581523895264, 3191.081166267395], 'efficientnet_b5-2': [62.55030632019043, 227.14972496032715], 'efficientnet_b5-4': [52.23727226257324, 122.29514122009276], 'efficientnet_b5-8': [48.195481300354, 137.98916339874268], 'efficientnet_b5-16': [71.48623466491699, 190.3698444366455], 'efficientnet_b5-32': [88.68300914764404, 229.1651964187622], 'efficientnet_b6-2': [49.16810989379883, 121.783447265625], 'efficientnet_b6-4': [72.48008251190186, 192.4673318862915], 'efficientnet_b6-8': [64.6524429321289, 187.1220111846924], 'efficientnet_b6-16': [59.7454309463501, 131.84561729431152], 'efficientnet_b6-32': [353.773832321167, 1243.6289310455325], 'efficientnet_b7-2': [68.09377670288086, 199.70269203186052], 'efficientnet_b7-4': [73.36020469665527, 187.29534149169928], 'efficientnet_b7-8': [99.68793392181396, 230.86111545562756], 'efficientnet_b7-16': [131.41703605651855, 542.326378822327], 'efficientnet_b7-32': [250.2284049987793, 584.7785949707031]}
    m_values = [row[f'm{i}'] for i in range(1, 4)]
    b_values = [row[f'b{i}'] for i in range(1, 4)]
    slo_values = [new_slos[f'{m}-{b}'] for m, b in zip(m_values, b_values)]
    return slo_values

def change_slos(job_size):
    df_250 = pd.read_csv('./data/raw-data/tb_250_data.csv')
    df_2400 = pd.read_csv('./data/raw-data/tb_2400_data.csv')

    df_250[['slo_m1', 'slo_m2', 'slo_m3']] = df_250.apply(get_new_slo, axis=1, result_type='expand')
    df_2400[['slo_m1', 'slo_m2', 'slo_m3']] = df_2400.apply(get_new_slo, axis=1, result_type='expand')

    for i in range(1, job_size + 1):
        df_250[f'slo_violation_m{i}'] = df_250.apply(lambda row: compute_slo_violation_new_slos(row, i), axis=1)
        df_2400[f'slo_violation_m{i}'] = df_2400.apply(lambda row: compute_slo_violation_new_slos(row, i), axis=1)

    slo_cols = [f'slo_violation_m{i}' for i in range(1, job_size + 1)]
    df_250['slo_violation_no'] = df_250[slo_cols].sum(axis=1)
    df_2400['slo_violation_no'] = df_2400[slo_cols].sum(axis=1)

    df_250.to_csv('./data/raw-data/tb_250_data_new_slos.csv', index=False)
    df_2400.to_csv('./data/raw-data/tb_2400_data_new_slos.csv', index=False)


if __name__=='__main__':

    '''
    This file is aggregating the raw data collected. Raw data collected is done in multiple steps:
    1. Running each job in isolation in closed loop, then in poisson distributino load 0.9. 
       We extract the p90 latency of the job in poisson isolation at load 0.9 as the SLO.
    2. Run each job mix of a given size on MPS. Obtain metrics (total, processed, queuing, tput).
    3. Run each job in MIG slices 1-4. This will be used to stitch together data per job mix.
    4. Profile each job using NVIDIA Nsight Compute and NVIDIA Nsight Systems. 

    We now aggregate the raw data into a csv file per job size. This file produces two rows per job mix: 
    one for MIG and one for MPS. 

    1. Aggregate MPS data - aggregate_mps()
    2. Add SLOs to data - add_slos_to_data()
    3. Aggregate MIG data - aggregate_mig_2()
    4. Stitch together mig data - stitch mig_data()
    5. Add SLOs to data - add_slos_to_data()
    6. Now save file - this file contains throughput and latency data per job mix per system. Naming convention
       for this file generated will be tiebreaker-size-{size}-metrics.csv
    
    Now we will use the tiebreaker-size-{size}-metrics.csv file to generate the training dataset for TieBreaker's 
    model. Steps are below:
    1. Label the data - label_data()
    2. Add profiling data - add_profiling_data()
    3. Drop columns not required for training data
    4. Now save file - this file doesn't contain throughput and latency data, it is the training data. Naming 
       convention for this file will be tiebreaker-training-set-size-{size}.csv

    '''


    size=3
    df = aggregate_mps(job_size=size)
    df = add_slos_to_data(df=df, job_size=size)
    print(df)
    # df.to_csv('./data/tb_first_250_data.csv', index=False)
    # print(df.to_string())
    # df_mig = aggregate_mig()
    # df_mig = aggregate_mig_2()
    # df_final = stitch_mig_data(df=df, df_mig=df_mig, job_size=size)
    # df_final = add_slos_to_data(df=df_final, job_size=size)
    # df_final.to_csv('./data/raw-data/tb_2400_data_size_2.csv', index=False)
    # print(df_final['job_mix'].value_counts().reset_index(name='count').to_string())
    # merge_data()
    # df = label_data()
    # df_final = add_profiling_data(df_init=df, job_size=3)
    # df_final.drop(['mode', 'load', 'slo_m1', 'slo_violation_m1', 'slo_m2', 'slo_violation_m2', 'slo_m3', 'slo_violation_m3', 'slo_violation_no', 'job_mix'], axis=1, inplace=True)
    # df_final.to_csv(f'./data/size-2650/tb-training-set-{size}.csv', index=False)

    # change_slos(3)
    # merge_data()

    # df = label_data(job_size=size)
    # df_final = add_profiling_data(df_init=df, job_size=size)
    # drop_list = [f'slo_m{i}' for i in range(1, size + 1)] + [f'slo_violation_m{i}' for i in range(1, size + 1)]
    # drop_list.extend(['mode', 'load', 'slo_violation_no', 'job_mix'])
    # df_final.drop(drop_list, axis=1, inplace=True)
    # df_final.to_csv(f'./data/size-2650/tb-training-set-{size}-new-slos.csv', index=False)




        
