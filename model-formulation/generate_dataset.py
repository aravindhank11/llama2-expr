import pandas as pd
import statistics
import os
import shutil

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

                            for mode in df_tput['mode'].unique():
                                data_point = []
                                # Get model and batch sizes of the job mix
                                job_mix = df_tput.columns.tolist()[2:]
                                for job in job_mix:
                                    data_point.append(job.split('-')[0][2:])
                                    data_point.append(int(job.split('-')[1]))

                                # Start expanding out data
                                data_point.extend(df_tput[df_tput['mode'] == mode].iloc[0])
                                # data_point.extend(df_process_p0[df_process_p0['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_process_p50[df_process_p50['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_process_p90[df_process_p90['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_process_p99[df_process_p99['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_process_p100[df_process_p100['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_total_p0[df_total_p0['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_total_p50[df_total_p50['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p90[df_total_p90['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p99[df_total_p99['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p100[df_total_p100['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_queued_p0[df_queued_p0['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_queued_p50[df_queued_p50['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_queued_p90[df_queued_p90['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_queued_p99[df_queued_p99['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                # data_point.extend(df_queued_p100[df_queued_p100['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_points.append(data_point)
                        else:
                            print(os.path.join(RESULTS_DIR, outer_dir, sub_dir))
    

    columns = []
    for i in range(1, job_size + 1):
        columns.append(f'm{i}')
        columns.append(f'b{i}')
    columns.append('mode')
    columns.append('load')
    metrics = ['tput', 'total_p90', 'total_p99', 'total_p100']
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
    
    columns = ['model', 'bs', 'mig_slice', 'tput', 'total_p0', 'total_p50', 'total_p90', 'total_p99', 'total_p100']
    df = pd.DataFrame(data_points, columns=columns)

    return df
    
def lookup_slo(row, job_no):
    MODEL_CONFIGS_250 = [['mobilenet_v2', 32, 23450.0], ['densenet121', 32, 59535.0], ['densenet201', 32, 94150.0], ['vgg19', 32, 77805.0], ['inception_v3', 32, 39270.0], ['resnet50', 32, 46585.0], ['resnet18', 32, 17955.0], ['resnet34', 32, 25375.0], ['resnet101', 32, 70035.0], ['resnet152', 32, 95725.0], ['densenet161', 32, 111125.0], ['densenet169', 32, 75495.0], ['vgg11', 32, 39620.0], ['vgg13', 32, 57400.0], ['vgg16', 32, 67620.0], ['alexnet', 32, 10920.0], ['squeezenet1_0', 32, 22785.0], ['squeezenet1_1', 32, 15995.0], ['mobilenet_v3_large', 32, 25025.0], ['mobilenet_v3_small', 32, 20230.0], ['efficientnet_b5', 32, 106645.0], ['efficientnet_b6', 32, 140455.0], ['efficientnet_b7', 32, 187285.0], ['mobilenet_v2', 2, 20650.0], ['densenet121', 2, 55545.0], ['densenet201', 2, 95025.0], ['vgg19', 2, 8855.0], ['inception_v3', 2, 40180.0], ['resnet50', 2, 24115.0], ['resnet18', 2, 8330.0], ['resnet34', 2, 13825.0], ['resnet101', 2, 43785.0], ['resnet152', 2, 63560.0], ['densenet161', 2, 78120.0], ['densenet169', 2, 78890.0], ['vgg11', 2, 5530.0], ['vgg13', 2, 6755.0], ['vgg16', 2, 7770.0], ['alexnet', 2, 3570.0], ['squeezenet1_0', 2, 9485.0], ['squeezenet1_1', 2, 8680.0], ['mobilenet_v3_large', 2, 24395.0], ['mobilenet_v3_small', 2, 20895.0], ['efficientnet_b5', 2, 74445.0], ['efficientnet_b6', 2, 85575.0], ['efficientnet_b7', 2, 102865.0], ['mobilenet_v2', 4, 20370.0], ['densenet121', 4, 55545.0], ['densenet201', 4, 95200.0], ['vgg19', 4, 13020.0], ['inception_v3', 4, 40425.0], ['resnet50', 4, 23625.0], ['resnet18', 4, 8295.0], ['resnet34', 4, 13720.0], ['resnet101', 4, 43785.0], ['resnet152', 4, 64085.0], ['densenet161', 4, 77210.0], ['densenet169', 4, 78855.0], ['vgg11', 4, 7560.0], ['vgg13', 4, 10010.0], ['vgg16', 4, 11515.0], ['alexnet', 4, 3570.0], ['squeezenet1_0', 4, 9590.0], ['squeezenet1_1', 4, 8820.0], ['mobilenet_v3_large', 4, 24605.0], ['mobilenet_v3_small', 4, 20755.0], ['efficientnet_b5', 4, 73149.99999999999], ['efficientnet_b6', 4, 85855.0], ['efficientnet_b7', 4, 107660.0], ['mobilenet_v2', 8, 20475.0], ['densenet121', 8, 55685.0], ['densenet201', 8, 95760.0], ['vgg19', 8, 23275.0], ['inception_v3', 8, 40880.0], ['resnet50', 8, 23975.0], ['resnet18', 8, 8960.0], ['resnet34', 8, 14035.0], ['resnet101', 8, 44135.0], ['resnet152', 8, 64155.0], ['densenet161', 8, 77105.0], ['densenet169', 8, 78680.0], ['vgg11', 8, 13055.0], ['vgg13', 8, 17815.0], ['vgg16', 8, 20475.0], ['alexnet', 8, 4585.0], ['squeezenet1_0', 8, 9765.0], ['squeezenet1_1', 8, 8855.0], ['mobilenet_v3_large', 8, 24710.0], ['mobilenet_v3_small', 8, 21385.0], ['efficientnet_b5', 8, 74760.0], ['efficientnet_b6', 8, 87815.0], ['efficientnet_b7', 8, 107310.0], ['mobilenet_v2', 16, 20440.0], ['densenet121', 16, 56980.0], ['densenet201', 16, 93695.0], ['vgg19', 16, 41370.0], ['inception_v3', 16, 40355.0], ['resnet50', 16, 26390.0], ['resnet18', 16, 10535.0], ['resnet34', 16, 15085.0], ['resnet101', 16, 43995.0], ['resnet152', 16, 64330.0], ['densenet161', 16, 76965.0], ['densenet169', 16, 78470.0], ['vgg11', 16, 21770.0], ['vgg13', 16, 30835.0], ['vgg16', 16, 36050.0], ['alexnet', 16, 6755.0], ['squeezenet1_0', 16, 13020.0], ['squeezenet1_1', 16, 9660.0], ['mobilenet_v3_large', 16, 25585.0], ['mobilenet_v3_small', 16, 21490.0], ['efficientnet_b5', 16, 78610.0], ['efficientnet_b6', 16, 90055.0], ['efficientnet_b7', 16, 110390.0]]

    slos = {}
    for item in MODEL_CONFIGS_250:
        slos[f'{item[0]}-{item[1]}'] = item[2]

    model_batch = str(row[f'm{job_no}']) + '-' + str(row[f'b{job_no}'])
    return slos[model_batch] / 1000

def compute_slo_violation(row, job_no):
    slo_violation = 0
    if row[f'slo_m{job_no}'] < row[f'total_p100_m{job_no}']:
        slo_violation = 1
    return slo_violation

def add_slos_to_data(df, job_size):
    df_slos = df.copy()
    for i in range(1, job_size + 1):
        df_slos[f'slo_m{i}'] = df_slos.apply(lambda row: lookup_slo(row, i), axis=1)
        df_slos[f'slo_violation_m{i}'] = df_slos.apply(lambda row: compute_slo_violation(row, i), axis=1)


    slo_cols = [f'slo_violation_m{i}' for i in range(1, job_size + 1)]
    df_slos['slo_violation_no'] = df_slos[slo_cols].sum(axis=1)
    # print(df_slos[['m1', 'b1', 'm2', 'b2', 'm3', 'b3', 'slo_m1', 'slo_m2', 'slo_m3']].to_string())
    return df_slos

def stitch_mig_data(df, df_mig, job_size):

    m1s = []
    b1s = []
    m2s = []
    b2s = []
    m3s = []
    b3s = []
    modes = []
    loads = []
    tput_m1s = []
    tput_m2s = []
    tput_m3s = []
    total_p90_m1s = []
    total_p90_m2s = []
    total_p90_m3s = []
    total_p99_m1s = []
    total_p99_m2s = []
    total_p99_m3s = []
    total_p100_m1s = []
    total_p100_m2s = []
    total_p100_m3s = []
    slo_m1s = []
    slo_violation_m1s = []
    slo_m2s = []
    slo_violation_m2s = []
    slo_m3s = []
    slo_violation_m3s = []
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

            m3 = split_line[2].split("-")[0]
            b3 = int(split_line[2].split("-")[1])
            slo_m3 = float(split_line[2].split("-")[2]) / 1000
            
            if len(df[(df['m1'] == m1) & (df['b1'] == b1) & (df['slo_m1'] == slo_m1) & (df['m2'] == m2) & (df['b2'] == b2) & (df['slo_m2'] == slo_m2) & (df['m3'] == m3) & (df['b3'] == b3) & (df['slo_m3'] == slo_m3)]) != 0:
                m1_slice = int(split_line[0].split("-")[3])
                m2_slice = int(split_line[1].split("-")[3])
                m3_slice = int(split_line[2].split("-")[3])

                m1s.append(m1)
                b1s.append(b1)
                m2s.append(m2)
                b2s.append(b2)
                m3s.append(m3)
                b3s.append(b3)
                modes.append('mig')
                loads.append(1.0)
                tput_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['tput'].iloc[0])
                tput_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['tput'].iloc[0])
                tput_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['tput'].iloc[0])
                total_p90_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p90'].iloc[0])
                total_p90_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p90'].iloc[0])
                total_p90_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p90'].iloc[0])
                total_p99_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p99'].iloc[0])
                total_p99_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p99'].iloc[0])
                total_p99_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p99'].iloc[0])
                total_p100_m1s.append(df_mig[(df_mig['model'] == m1) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p100'].iloc[0])
                total_p100_m2s.append(df_mig[(df_mig['model'] == m2) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p100'].iloc[0])
                total_p100_m3s.append(df_mig[(df_mig['model'] == m3) & (df_mig['bs'] == b1) & (df_mig['mig_slice'] == m1_slice)]['total_p100'].iloc[0])
                slo_m1s.append(-1)
                slo_m2s.append(-1)
                slo_m3s.append(-1)
                slo_violation_m1s.append(-1)
                slo_violation_m2s.append(-1)
                slo_violation_m3s.append(-1)
                slo_violation_nos.append(-1)
            else:
                break
        
    data = {'m1': m1s, 'b1': b1s, 'm2': m2s, 'b2': b2s, 'm3': m3s, 'b3': b3s, 'mode': modes, 'load': loads, 'tput_m1': tput_m1s, 'tput_m2': tput_m2s, 'tput_m3': tput_m3s, 'total_p90_m1': total_p90_m1s, 'total_p90_m2': total_p90_m2s, 'total_p90_m3': total_p90_m3s,
            'total_p99_m1': total_p99_m1s, 'total_p99_m2': total_p99_m2s, 'total_p99_m3': total_p99_m3s, 'total_p100_m1': total_p100_m1s, 'total_p100_m2': total_p100_m2s, 'total_p100_m3': total_p100_m3s, 'slo_m1': slo_m1s, 'slo_violation_m1': slo_violation_m1s,
            'slo_m2': slo_m2s, 'slo_violation_m2': slo_violation_m2s, 'slo_m3': slo_m3s, 'slo_violation_m3': slo_violation_m3s, 'slo_violation_no': slo_violation_nos}

    df_mig_add = pd.DataFrame(data)
    df_final = pd.concat([df, df_mig_add])
    df_final['job_mix'] = df_final.apply(lambda row: '-'.join([str(row['m1']), str(row['b1']), str(row['m2']), str(row['b2']), str(row['m3']), str(row['b3'])]), axis=1)

    return df_final

def merge_data():
    df1 = pd.read_csv('./data/tb_250_data.csv')
    df2 = pd.read_csv('./data/tb_2400_data.csv')

    df1['job_mix'] = df1.apply(lambda row: '-'.join([str(row['m1']), str(row['b1']), str(row['m2']), str(row['b2']), str(row['m3']), str(row['b3'])]), axis=1)
    df_final = pd.concat([df1, df2])
    df_final.to_csv('./data/tb-large-training-set-3.csv', index=False)

def label_data():
    # TODO: make agnostic to job size
    df = pd.read_csv('./data/raw-data/tb-large-training-set-3.csv')
    labels = {}
    data_points = []

    for job_mix in df['job_mix'].unique():
        df_mps = df[(df['job_mix'] == job_mix) & (df['mode'] == 'mps-uncap')]
        df_mig = df[(df['job_mix'] == job_mix) & (df['mode'] == 'mig')]
        mps_tput = df_mps[['tput_m1', 'tput_m2', 'tput_m3']].sum().sum()
        mps_tail_latency = df_mps[['total_p100_m1', 'total_p100_m2', 'total_p100_m3']].max().max()
        mps_slo_violations = df_mps['slo_violation_no'].tolist()[0]
        mig_tput = df_mig[['tput_m1', 'tput_m2', 'tput_m3']].sum().sum()
        mig_tail_latency = df_mig[['total_p100_m1', 'total_p100_m2', 'total_p100_m3']].max().max()
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

if __name__=='__main__':
    size=3
    # df = aggregate_mps(job_size=3)
    # df = add_slos_to_data(df=df, job_size=3)
    # df.to_csv('./data/tb_first_250_data.csv', index=False)
    # print(df.to_string())
    # df_mig = aggregate_mig()
    # df_final = stitch_mig_data(df=df, df_mig=df_mig, job_size=3)
    # df_final = add_slos_to_data(df=df_final, job_size=3)
    # print(df_final)
    # df_final.to_csv('./data/tb_2400_data.csv', index=False)
    # print(df_final['job_mix'].value_counts().reset_index(name='count').to_string())
    # merge_data()
    df = label_data()
    df_final = add_profiling_data(df_init=df, job_size=3)
    df_final.drop(['mode', 'load', 'slo_m1', 'slo_violation_m1', 'slo_m2', 'slo_violation_m2', 'slo_m3', 'slo_violation_m3', 'slo_violation_no', 'job_mix'], axis=1, inplace=True)
    df_final.to_csv(f'./data/size-2650/tb-training-set-{size}.csv', index=False)



        
