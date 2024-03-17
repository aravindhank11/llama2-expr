import pandas as pd
import sqlite3
import statistics


DATA_STORE = '/home/ps35324/gpu-sharing-scheduler/src/data-store/scheduler.db'
PROFILE_DB = '../profiler/data/a100/model_profiles.csv'
MAX_SLOWDOWN = 4

def create_df(pk_value):

    conn = sqlite3.connect(DATA_STORE)
    curr = conn.cursor()

    sql_data = 'SELECT * FROM experiment_results WHERE query_id > {}'.format(pk_value)
    df_results = pd.read_sql_query(sql_data, conn)

    # Get duration of critical section and e2e of each inference
    df_results['time_infer_end'] = pd.to_datetime(df_results['time_infer_end'])
    df_results['time_infer_start'] = pd.to_datetime(df_results['time_infer_start'])
    df_results['time_q_received'] = pd.to_datetime(df_results['time_q_received'])
    df_results['time_q_responded'] = pd.to_datetime(df_results['time_q_responded'])
    df_results['cs_duration'] = (df_results['time_infer_end'] - df_results['time_infer_start']).dt.total_seconds().mul(1000).astype(float)
    df_results['e2e_duration'] = (df_results['time_q_responded'] - df_results['time_q_received']).dt.total_seconds().mul(1000).astype(float)

    # Update SLO information
    df_results['slo_ms'] = df_results['slo'].div(1000).astype(float)
    df_results['slo_violation'] = df_results['cs_duration'] > df_results['slo_ms']
    df_results['slo_violation'] = df_results['slo_violation'].astype(int)

    # Get execution time of one request on MPS with no concurrency in milliseconds - per model/batchsize combo
    cs_dur_query = 'SELECT model_name, max_batch_size, e2e_exec FROM model_profiles WHERE model_id > 18'
    df_cs_dur = pd.read_sql_query(cs_dur_query, conn)
    df_cs_dur['e2e_exec_ms'] = df_cs_dur['e2e_exec'] / 1000
    df_cs_dur['max_batch_size'] = df_cs_dur['max_batch_size'].apply(lambda x: str(x))
    df_cs_dur['name_bs'] = df_cs_dur['model_name'] + '_' + df_cs_dur['max_batch_size']
    cs_dur_dict = pd.Series(df_cs_dur.e2e_exec_ms.values,index=df_cs_dur.name_bs).to_dict()

    df_results['cs_duration'] = df_results['cs_duration'].div(100).astype(float)
    df_results['slowdown'] = df_results.apply(lambda row: row['cs_duration'] / cs_dur_dict[row['model_name'] + '_' + str(row['batch_size'])], axis=1)

    # Remove combinations where MIG slowdown is greater than 4
    removal_list = []
    for num in [2, 3, 4, 5, 6, 7]:
        sub_df = df_results[(df_results['slowdown'] > 3.8) & (df_results['system'] == 'multi-bs-mig-{}'.format(num))]
        for exp_no in sub_df['exp_no'].unique():
            new_list = []
            gpu_1 = False
            gpu_2 = False
            sub_df_exp = sub_df[sub_df['exp_no'] == exp_no]
            for exp_model_no in sub_df_exp['exp_model_no'].unique():
                if exp_model_no <= num:
                    gpu_1 = True
                if exp_model_no > num:
                    gpu_2 = True
            new_list.append(exp_no)
            new_list.append(num)
            new_list.append(gpu_1)
            new_list.append(gpu_2)
            removal_list.append(new_list)
    
    return df_results, removal_list

def create_initial_dataset(job_size):
    df, removal_list = create_df(187028)
    combos = []
    # Determine best concurrency mechanism for each model cobmination
    top_max_mechanisms = []
    top_median_mechanisms = []
    max_slowdowns = []
    median_slowdowns = []

    # Get all runs of the given job size on MPS, TM, and MIG
    df_job_size = df[(df['system'] == 'multi-bs-mps-{}'.format(job_size)) | (df['system'] == 'multi-bs-ts-{}'.format(job_size)) | (df['system'] == 'multi-bs-mig-{}'.format(job_size))]

    for exp_no in df_job_size['exp_no'].unique():
        # Get job mix ran on each GPU
        df_exp_1 = df_job_size[(df_job_size['exp_no'] == exp_no) & (df_job_size['exp_model_no'] <= job_size)]
        df_exp_2 = df_job_size[(df_job_size['exp_no'] == exp_no) & (df_job_size['exp_model_no'] > job_size)]

        exp_1_max_dict = {}
        exp_2_max_dict = {}
        exp_1_median_dict = {}
        exp_2_median_dict = {}

        best_mech_max_1 = ''
        best_mech_max_2 = ''
        best_mech_median_1 = ''
        best_mech_median_2 = ''

        for mechanism in ['mps', 'ts', 'mig']:
            sub_df_1 = df_exp_1[df_exp_1['system'] == 'multi-bs-{}-{}'.format(mechanism, job_size)]
            sub_df_2 = df_exp_2[df_exp_2['system'] == 'multi-bs-{}-{}'.format(mechanism, job_size)]

            # Get the row with the max slowdown for each system
            max_row_1 = sub_df_1.loc[sub_df_1['slowdown'].idxmax()]
            max_row_2 = sub_df_2.loc[sub_df_2['slowdown'].idxmax()]
            exp_1_max_dict[max_row_1['system']] = max_row_1['slowdown']
            exp_2_max_dict[max_row_2['system']] = max_row_2['slowdown']

            # Get the row with the median slowdown for each system
            median_row_1 = sub_df_1.loc[sub_df_1['slowdown'] == statistics.median_low(sub_df_1['slowdown'].tolist())]
            median_row_2 = sub_df_2.loc[sub_df_2['slowdown'] == statistics.median_low(sub_df_2['slowdown'].tolist())]

            exp_1_median_dict[median_row_1['system'].tolist()[0]] = median_row_1['slowdown'].tolist()[0]
            exp_2_median_dict[median_row_2['system'].tolist()[0]] = median_row_2['slowdown'].tolist()[0]

            # Label is the system with the minimum max slowdown
            best_mech_max_1 = min(exp_1_max_dict, key = lambda x: exp_1_max_dict[x])
            best_mech_max_2 = min(exp_2_max_dict, key = lambda x: exp_2_max_dict[x])

            # Label is the system with the minimum median slowdown
            best_mech_median_1 = min(exp_1_median_dict, key = lambda x: exp_1_median_dict[x])
            best_mech_median_2 = min(exp_2_median_dict, key = lambda x: exp_2_median_dict[x])

        top_max_mechanisms.append(best_mech_max_1)
        top_max_mechanisms.append(best_mech_max_2)
        top_median_mechanisms.append(best_mech_median_1)
        top_median_mechanisms.append(best_mech_median_2)
        max_slowdowns.append(exp_1_max_dict)
        max_slowdowns.append(exp_2_max_dict)
        median_slowdowns.append(exp_1_median_dict)
        median_slowdowns.append(exp_2_median_dict)
    
    # Associate best mechanisms with each mode combo
    index = 0
    for exp_no in df_job_size['exp_no'].unique():
        df_exp = df_job_size[df_job_size['exp_no'] == exp_no]
        combo_1 = []
        combo_2 = []
        for exp_model_no in df_exp['exp_model_no'].unique():
            df_model = df_exp[df_exp['exp_model_no'] == exp_model_no]
            model = list(df_model['model_name'].unique())[0]
            bs = list(df_model['batch_size'].unique())[0]
            if exp_model_no <= job_size:
                # combo_1.append(model + '_' + str(bs))
                combo_1.append(model)
                combo_1.append(bs)
            else:
                # combo_2.append(model + '_' + str(bs))
                combo_2.append(model)
                combo_2.append(bs)
        
        combo_1.append(top_max_mechanisms[index].split('-')[2])
        combo_2.append(top_max_mechanisms[index + 1].split('-')[2])
        combo_1.append(top_median_mechanisms[index].split('-')[2])
        combo_2.append(top_median_mechanisms[index + 1].split('-')[2])

        for mechanism in ['mps', 'ts', 'mig']:
            combo_1.append(max_slowdowns[index]['multi-bs-{}-{}'.format(mechanism, job_size)])
            combo_2.append(max_slowdowns[index+1]['multi-bs-{}-{}'.format(mechanism, job_size)])
            combo_1.append(median_slowdowns[index]['multi-bs-{}-{}'.format(mechanism, job_size)])
            combo_2.append(median_slowdowns[index+1]['multi-bs-{}-{}'.format(mechanism, job_size)])
        combo_1.append(str(exp_no) + '_1')
        combo_2.append(str(exp_no) + '_2')
        # if len(combo_1) == 15:
        #     print(combo_1)
        # if len(combo_2) == 15:
        #     print(combo_2)
        combos.append(combo_1)
        combos.append(combo_2)
        index += 2

    labels = []
    for i in range(1, job_size + 1):
        labels.append(f'm_{i}')
        labels.append(f'bs_{i}')
    labels = labels + ['top_max_mech', 'top_median_mech', 'mps_max_slowdown', 'mps_median_slowdown', 'ts_max_slowdown', 'ts_median_slowdown', 'mig_max_slowdown', 'mig_median_slowdown', 'exp_gpu_no']

    df_init = pd.DataFrame(combos, columns=labels)

    for l in removal_list:
        df_drop = ''
        if l[2] == True:
            df_drop = df_init[(df_init['exp_gpu_no'] == (str(l[0]) + '_1'))]
        elif l[3] == True:
            df_drop = df_init[(df_init['exp_gpu_no'] == (str(l[0]) + '_2'))]
        df_init.drop(df_drop.index, inplace=True)
    
    return df_init

def create_initial_dataset_2(job_size):
    df, removal_list = create_df(187028)
    combos = []

    # Get all runs of the given job size on MPS
    df_job_size = df[(df['system'] == 'multi-bs-mps-{}'.format(job_size))]

    for exp_no in df_job_size['exp_no'].unique():
        # Get job mix ran on each GPU
        df_exp_1 = df_job_size[(df_job_size['exp_no'] == exp_no) & (df_job_size['exp_model_no'] <= job_size)]
        df_exp_2 = df_job_size[(df_job_size['exp_no'] == exp_no) & (df_job_size['exp_model_no'] > job_size)]

        # Check if max slowdown is within reason
        label_1 = 'no_hol'
        label_2 = 'no_hol'
        if df_exp_1['slowdown'].max() > 4:
            label_1 = 'hol'
        if df_exp_2['slowdown'].max() > 4:
            label_2 = 'hol'
        
        combo = []
        for exp_model_no in df_exp_1['exp_model_no'].unique():
            df_model = df_exp_1[df_exp_1['exp_model_no'] == exp_model_no]
            model = list(df_model['model_name'].unique())[0]
            bs = list(df_model['batch_size'].unique())[0]
            combo.append(model)
            combo.append(bs)
        combo.append(label_1)
        combo.append(f'{exp_no}_1')
        combos.append(combo)

        combo = []
        for exp_model_no in df_exp_2['exp_model_no'].unique():
            df_model = df_exp_2[df_exp_2['exp_model_no'] == exp_model_no]
            model = list(df_model['model_name'].unique())[0]
            bs = list(df_model['batch_size'].unique())[0]
            combo.append(model)
            combo.append(bs)
        combo.append(label_2)
        combo.append(f'{exp_no}_2')
        combos.append(combo)

    labels = []
    for i in range(1, job_size + 1):
        labels.append(f'm_{i}')
        labels.append(f'bs_{i}')
    labels.append('label')
    labels.append('exp_gpu_no')

    df_init = pd.DataFrame(combos, columns=labels)

    for l in removal_list:
        df_drop = ''
        if l[2] == True:
            df_drop = df_init[(df_init['exp_gpu_no'] == (str(l[0]) + '_1'))]
        elif l[3] == True:
            df_drop = df_init[(df_init['exp_gpu_no'] == (str(l[0]) + '_2'))]
        df_init.drop(df_drop.index, inplace=True)
    
    return df_init



def add_profiling_data(df_init, job_size):
    df_profiled = pd.read_csv(PROFILE_DB)
    df_final = df_init
    for i in range(1, job_size + 1):
        df_final = pd.merge(df_final, df_profiled, left_on=[f'm_{i}', f'bs_{i}'], right_on=['model', 'batch_size'], how='left')
        new_column_names = {col: col + f'_{i}' for col in df_profiled.columns}
        df_final.rename(columns=new_column_names, inplace=True)
        df_final.drop([f'model_{i}', f'batch_size_{i}'], axis=1, inplace=True)
    return df_final

if __name__=='__main__':
    # for size in [2, 3, 4]:
    #     df = create_initial_dataset(job_size=size)
    #     df = add_profiling_data(df_init=df, job_size=size)
    #     df.to_csv(f'./data/tiebreaker-training-set-{size}.csv', index=False)

    for size in [2, 3, 4]:
        df = create_initial_dataset_2(job_size=size)
        df = add_profiling_data(df_init=df, job_size=size)
        df.to_csv(f'./data/tiebreaker-training-set-{size}-hol.csv', index=False)

