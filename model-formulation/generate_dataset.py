import pandas as pd
import statistics
import os
import shutil

RESULTS_DIR = '../results/a100'

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
                            for mode in df_tput['mode'].unique():
                                # Get model and batch sizes of the job mix
                                job_mix = df_tput.columns.tolist()[2:]
                                for job in job_mix:
                                    data_point.append(job.split('-')[0][2:])
                                    data_point.append(int(job.split('-')[1]))

                                # Start expanding out data
                                data_point.extend(df_tput[df_tput['mode'] == mode].iloc[0])
                                data_point.extend(df_process_p0[df_process_p0['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p50[df_process_p50['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p90[df_process_p90['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p99[df_process_p99['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_process_p100[df_process_p100['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p0[df_total_p0['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p50[df_total_p50['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p90[df_total_p90['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p99[df_total_p99['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_total_p100[df_total_p100['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p0[df_queued_p0['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p50[df_queued_p50['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p90[df_queued_p90['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p99[df_queued_p99['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_point.extend(df_queued_p100[df_queued_p100['mode'] == mode].iloc[0,-int(f'{job_size}'):])
                                data_points.append(data_point)
                        else:
                            print(os.path.join(RESULTS_DIR, outer_dir, sub_dir))

    columns = []
    for i in range(1, job_size + 1):
        columns.append(f'm_{i}')
        columns.append(f'b_{i}')
    columns.append('mode')
    columns.append('load')
    metrics = ['tput', 'process_p0', 'process_p50', 'process_p90', 'process_p99', 'process_p100', 'total_p0', 'total_p50', 'total_p90', 'total_p99', 'total_p100', 'queue_p0', 'queue_p50', 'queue_p90', 'queue_p99', 'queue_p100']
    for metric in metrics:
        for i in range(1, job_size + 1):
            columns.append(f'{metric}_m_{i}')

    df = pd.DataFrame(data_points, columns=columns)
    return df

def aggregate_mig():

    # Missing 2 models -- 'inception_v3', 'mobilenet_v2' 'mobilenet_v3_small' 'mobilenet_v3_large' 'densenet121' 'densenet161' 'densenet169'

    # models=('mobilenet_v2' 'mobilenet_v3_small' 'mobilenet_v3_large' 'densenet121' 'densenet161' 'densenet169')
    # models=('densenet201' 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'inception_v3')
    # models=('resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152' 'alexnet')
    # models=('squeezenet1_0' 'squeezenet1_1' 'efficientnet_b5' 'efficientnet_b6' 'efficientnet_b7')
    # batches=(2 4 8 16 32)


    for outer_dir in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, outer_dir)):
            models = outer_dir.split('-')
            # print(models)
            if len(models) == 2:
                print(os.path.join(RESULTS_DIR, outer_dir))
                # shutil.rmtree(os.path.join(RESULTS_DIR, outer_dir))
    

def lookup_slo(row, job_no):
    slos = {'mobilenet_v2-32': 23450.0, 'densenet121-32': 59535.0, 'densenet201-32': 94150.0, 'vgg19-32': 77805.0, 'inception_v3-32': 39270.0, 'resnet50-32': 46585.0, 'resnet18-32': 17955.0, 'resnet34-32': 25375.0, 'resnet101-32': 70035.0, 'resnet152-32': 95725.0, 'densenet161-32': 111125.0, 'densenet169-32': 75495.0, 'vgg11-32': 39620.0, 'vgg13-32': 57400.0, 'vgg16-32': 67620.0, 'alexnet-32': 10920.0, 'squeezenet1_0-32': 22785.0, 'squeezenet1_1-32': 15995.0, 'mobilenet_v3_large-32': 25025.0, 'mobilenet_v3_small-32': 20230.0, 'efficientnet_b5-32': 106645.0, 'efficientnet_b6-32': 140455.0, 'efficientnet_b7-32': 187285.0, 'mobilenet_v2-2': 0650.0, 'densenet121-2': 5545.0, 'densenet201-2': 5025.0, 'vgg19-2': 855.0, 'inception_v3-2': 0180.0, 'resnet50-2': 4115.0, 'resnet18-2': 330.0, 'resnet34-2': 3825.0, 'resnet101-2': 3785.0, 'resnet152-2': 3560.0, 'densenet161-2': 8120.0, 'densenet169-2': 8890.0, 'vgg11-2': 530.0, 'vgg13-2': 755.0, 'vgg16-2': 770.0, 'alexnet-2': 570.0, 'squeezenet1_0-2': 485.0, 'squeezenet1_1-2': 680.0, 'mobilenet_v3_large-2': 4395.0, 'mobilenet_v3_small-2': 0895.0, 'efficientnet_b5-2': 4445.0, 'efficientnet_b6-2': 5575.0, 'efficientnet_b7-2': 02865.0, 'mobilenet_v2-4': 0370.0, 'densenet121-4': 5545.0, 'densenet201-4': 5200.0, 'vgg19-4': 3020.0, 'inception_v3-4': 0425.0, 'resnet50-4': 3625.0, 'resnet18-4': 295.0, 'resnet34-4': 3720.0, 'resnet101-4': 3785.0, 'resnet152-4': 4085.0, 'densenet161-4': 7210.0, 'densenet169-4': 8855.0, 'vgg11-4': 560.0, 'vgg13-4': 0010.0, 'vgg16-4': 1515.0, 'alexnet-4': 570.0, 'squeezenet1_0-4': 590.0, 'squeezenet1_1-4': 820.0, 'mobilenet_v3_large-4': 4605.0, 'mobilenet_v3_small-4': 0755.0, 'efficientnet_b5-4': 3149.99999999999, 'efficientnet_b6-4': 5855.0, 'efficientnet_b7-4': 07660.0, 'mobilenet_v2-8': 0475.0, 'densenet121-8': 5685.0, 'densenet201-8': 5760.0, 'vgg19-8': 3275.0, 'inception_v3-8': 0880.0, 'resnet50-8': 3975.0, 'resnet18-8': 960.0, 'resnet34-8': 4035.0, 'resnet101-8': 4135.0, 'resnet152-8': 4155.0, 'densenet161-8': 7105.0, 'densenet169-8': 8680.0, 'vgg11-8': 3055.0, 'vgg13-8': 7815.0, 'vgg16-8': 0475.0, 'alexnet-8': 585.0, 'squeezenet1_0-8': 765.0, 'squeezenet1_1-8': 855.0, 'mobilenet_v3_large-8': 4710.0, 'mobilenet_v3_small-8': 1385.0, 'efficientnet_b5-8': 4760.0, 'efficientnet_b6-8': 7815.0, 'efficientnet_b7-8': 07310.0, 'mobilenet_v2-16': 20440.0, 'densenet121-16': 56980.0, 'densenet201-16': 93695.0, 'vgg19-16': 41370.0, 'inception_v3-16': 40355.0, 'resnet50-16': 26390.0, 'resnet18-16': 10535.0, 'resnet34-16': 15085.0, 'resnet101-16': 43995.0, 'resnet152-16': 64330.0, 'densenet161-16': 76965.0, 'densenet169-16': 78470.0, 'vgg11-16': 21770.0, 'vgg13-16': 30835.0, 'vgg16-16': 36050.0, 'alexnet-16': 6755.0, 'squeezenet1_0-16': 13020.0, 'squeezenet1_1-16': 9660.0, 'mobilenet_v3_large-16': 25585.0, 'mobilenet_v3_small-16': 21490.0, 'efficientnet_b5-16': 78610.0, 'efficientnet_b6-16': 90055.0, 'efficientnet_b7-16': 110390.0}
    model_batch = str(row[f'm_{job_no}']) + '-' + str(row[f'b_{job_no}'])
    return slos[model_batch]

def add_slos_to_data(df, job_size):
    df_slos = df.copy()
    for i in range(1, job_size + 1):
        df_slos[f'slo_m_{i}'] = df_slos.apply(lambda row: lookup_slo(row, i), axis=1)
    print(df_slos)
    return df_slos
        
# def add_mig_data_points():


if __name__=='__main__':
    # df = aggregate_mps(job_size=3)
    # df = add_slos_to_data(df=df, job_size=3)
    aggregate_mig()