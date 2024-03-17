import pandas as pd
import sqlite3
import argparse
import os
from math import ceil, floor

def view_processed_ncu_data(profile_dir, model, batch_size):
    ncu_path = profile_dir + f'batchsize_{batch_size}_output_ncu_processed.csv'
    df = pd.read_csv(ncu_path)

    print(df[['Elapsed Cycles', 'DRAM Throughput', 'Duration', 'L1/TEX Cache Throughput', 'L2 Cache Throughput', 'SM Active Cycles', 'Compute (SM) Throughput', 'Executed Ipc Active', 'Executed Ipc Elapsed', 'Issue Slots Busy', 'Issued Ipc Active', 'SM Busy']].to_string())
    # print(df[['L1/TEX Hit Rate', 'L2 Hit Rate', 'Block Size', 'Grid Size', 'Registers Per Thread', 'Shared Memory Configuration Size', 'Driver Shared Memory Per Block', 'Dynamic Shared Memory Per Block', 'Static Shared Memory Per Block', 'Threads', 'Waves Per SM']].to_string())
    # print(df[['Block Limit SM', 'Block Limit Registers', 'Block Limit Shared Mem', 'Block Limit Warps', 'Theoretical Active Warps per SM', 'Theoretical Occupancy', 'Achieved Occupancy', 'Achieved Active Warps Per SM', 'Branch Instructions Ratio', 'Branch Instructions', 'Branch Efficiency', 'Avg. Divergent Branches']].to_string())

def aggregate_raw_ncu_data(profile_dir, model, batch_size):
    ncu_path = profile_dir + f'batchsize_{batch_size}_output_ncu_processed.csv'
    df = pd.read_csv(ncu_path)

    final_data = {}
    for col in df.columns:
        if col != 'kern_name':
            data = {'min': df[col].min(),
                    'p10': df[col].quantile(0.1),
                    'p25': df[col].quantile(0.25),
                    'median': df[col].median(),
                    'mean': df[col].mean(),
                    'p75': df[col].quantile(0.75),
                    'p90': df[col].quantile(0.9),
                    'p95': df[col].quantile(0.95),
                    'p99': df[col].quantile(0.99),
                    'max': df[col].max(),
                    'variance': df[col].var()
                    }
            for key in data:
                final_data[f'{col}_{key}'] = [data[key]]
                if col == 'elapsed_cycles' or col == 'kern_duration':
                    final_data[f'{col}_total'] = df[col].sum()
    df_final = pd.DataFrame(final_data)
    df_final['num_kerns'] = len(df)
    return df_final

def compute_sms_required(df):
    
    max_threads_sm = 2048
    thread_per_warp = 32
    registers_per_sm = 65536

    sm_needed = []
    for index, row in df.iterrows():
        num_blocks = row['grid_size']
        num_threads = row['threads']
        threads_per_block = row['block_size']
        shmem_per_block = row['static_shmem_per_block']
        regs_per_thread = row['reg_per_thread']
        configured_shmem_size = row['configured_shmem_size']

        # from threads
        blocks_per_sm_threads = ceil(max_threads_sm/threads_per_block)

        # from shmem
        blocks_per_sm_shmem = blocks_per_sm_threads
        if shmem_per_block > 0:
            blocks_per_sm_shmem = ceil(configured_shmem_size/shmem_per_block)
        
        # from registers
        regs_per_warp = ceil(thread_per_warp*regs_per_thread/256) * 256
        warps_per_sm = floor((registers_per_sm/regs_per_warp)/4) * 4
        warps_per_block = ceil(threads_per_block/32)
        blocks_per_sm_regs = int(warps_per_sm/warps_per_block)

        blocks_per_sm = min(blocks_per_sm_threads, blocks_per_sm_shmem, blocks_per_sm_regs)
        sm_needed_kernel = ceil(num_blocks/blocks_per_sm)
        sm_needed.append(sm_needed_kernel)
    return sm_needed

def process_raw_ncu_data(profile_dir, model, batch_size):
    ncu_path = profile_dir + f'batchsize_{batch_size}_output_ncu.csv'
    df = pd.read_csv(ncu_path, index_col=0)

    kernels = []
    # 35 metrics
    # metrics_to_get = [
    #     'Elapsed Cycles',
    #     'DRAM Throughput',
    #     'Duration',
    #     'L1/TEX Cache Throughput',
    #     'L2 Cache Throughput',
    #     'SM Active Cycles',
    #     'Compute (SM) Throughput',
    #     'Executed Ipc Active',
    #     'Executed Ipc Elapsed',
    #     'Issue Slots Busy',
    #     'Issued Ipc Active',
    #     'SM Busy',
    #     'L1/TEX Hit Rate',
    #     'L2 Hit Rate',
    #     'Block Size',
    #     'Grid Size',
    #     'Registers Per Thread',
    #     'Shared Memory Configuration Size',
    #     'Driver Shared Memory Per Block',
    #     'Dynamic Shared Memory Per Block',
    #     'Static Shared Memory Per Block',
    #     'Threads',
    #     'Waves Per SM',
    #     'Block Limit SM',
    #     'Block Limit Registers',
    #     'Block Limit Shared Mem',
    #     'Block Limit Warps',
    #     'Theoretical Active Warps per SM',
    #     'Theoretical Occupancy',
    #     'Achieved Occupancy',
    #     'Achieved Active Warps Per SM',
    #     'Branch Instructions Ratio',
    #     'Branch Instructions',
    #     'Branch Efficiency',
    #     'Avg. Divergent Branches'        
    # ]

    # 12 metrics
    metrics_to_get = [
        'DRAM Throughput',
        'Duration',
        'L2 Cache Throughput',
        'Compute (SM) Throughput',
        'SM Busy',
        'Block Size',
        'Grid Size',
        'Registers Per Thread',
        'Shared Memory Configuration Size',
        'Static Shared Memory Per Block',
        'Threads',
        'Theoretical Occupancy',
        'Achieved Occupancy',
    ]

    unique_kernel_names = set()

    for index, row in df.iterrows():
        kernel = row['Kernel Name']
        metric_name = row['Metric Name']

        if metric_name == 'DRAM Frequency':
            kernels.append([kernel])
            unique_kernel_names.add(kernel)
        elif metric_name in metrics_to_get:
            kernels[-1].append(float(row['Metric Value'].replace(",","")))

    labels = ['kern_name', 'dram_tput', 'kern_duration', 'l2_cache_tput', 'compute_tput', 
              'sm_busy', 'block_size', 'grid_size', 'reg_per_thread', 'configured_shmem_size', 'static_shmem_per_block',
              'threads', 'theoretical_occupancy', 'achieved_occupancy']
    # print(kernels)
    # labels = ['kern_name', 'elapsed_cycles', 'dram_tput', 'kern_duration', 'l1_cache_tput', 'l2_cache_tput', 'sm_active_cycles', 'compute_tput', 
    #           'executed_ipc_active', 'executed_ipc_elapsed', 'issue_slots_busy', 'issued_ipc_active', 'sm_busy', 'l1_hit_rate', 'l2_hit_rate', 
    #           'block_size', 'grid_size', 'reg_per_thread', 'shared_mem_config_size', 'driver_shared_mem_per_block', 'dynamic_shared_mem_per_block',
    #           'static_shared_mem_per_block', 'threads', 'waves_per_sm', 'block_limit_sm', 'block_limig_registers', 'block_limit_shared_mem', 'block_limit_warps',
    #           'theoretical_active_warps_per_sm', 'theoretical_occupancy', 'achieved_occupancy', 'achieved_active_warps_per_sm', 
    #           'branch_instr_ratio', 'branch_instr', 'branch_efficiency', 'avg_divergent_branches']

    df_new = pd.DataFrame(kernels, columns=labels)
    df_new['sm_needed'] = compute_sms_required(df_new)
    df_new.to_csv(profile_dir + f'batchsize_{batch_size}_output_ncu_processed.csv', index=False)

def extract_nsys_data(profile_dir, model, batch_size):
    
    # Connect to SQLITE DB created by nsys profiler
    nsys_sqlite_path = profile_dir + f'batchsize_{batch_size}_output_nsys.sqlite'
    conn = None
    try:
        conn = sqlite3.connect((nsys_sqlite_path))
    except Error as e:
        print(f'Could not connect to generated nys database at {nsys_sqlite_path}')
        return -1
    
    # SQLITE commands
    memcpy_cmd = '''
                    SELECT CUPTI_ACTIVITY_KIND_RUNTIME.start AS async_start,
                    CUPTI_ACTIVITY_KIND_RUNTIME.end AS async_end,
                    CUPTI_ACTIVITY_KIND_MEMCPY.start AS memcpy_start,
                    CUPTI_ACTIVITY_KIND_MEMCPY.end AS memcpy_end,
                    CUPTI_ACTIVITY_KIND_MEMCPY.bytes AS memcpy_bytes
                    FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    JOIN CUPTI_ACTIVITY_KIND_RUNTIME ON CUPTI_ACTIVITY_KIND_MEMCPY.correlationID = CUPTI_ACTIVITY_KIND_RUNTIME.correlationID
                    WHERE CUPTI_ACTIVITY_KIND_MEMCPY.copyKind = 1
                '''
    kernel_cmd = 'SELECT start, end from CUPTI_ACTIVITY_KIND_KERNEL'

    # Get data into dataframe
    df_memcpy = pd.read_sql_query(memcpy_cmd, conn)
    df_kernel = pd.read_sql_query(kernel_cmd, conn)
    # print(df_kernel.to_string())
    conn.close()

    # print(df_memcpy.to_string())

    # Get memcpy information for model load
    load_time = float(df_memcpy.iloc[-1]['memcpy_end']) - float(df_memcpy.iloc[-1]['memcpy_start'])
    model_size = int(sum(df_memcpy['memcpy_bytes']))

    # Get kernel execution data only for last request
    df_memcpy_request = df_memcpy.iloc[-1]
    image_load_time = df_memcpy_request['memcpy_end'] - df_memcpy_request['memcpy_start']
    df_request_kernel = df_kernel[df_kernel['start'] >= df_memcpy_request['memcpy_end']].copy()
    df_request_kernel['duration'] = df_request_kernel['end'] - df_request_kernel['start']
    # print(df_request_kernel.to_string())

    kernel_min_dur = float(df_request_kernel['duration'].min())
    kernel_p10_dur = float(df_request_kernel['duration'].quantile(0.1))
    kernel_p25_dur = float(df_request_kernel['duration'].quantile(0.25))
    kernel_mean_dur = float(df_request_kernel['duration'].mean())
    kernel_median_dur = float(df_request_kernel['duration'].median())
    kernel_p75_dur = float(df_request_kernel['duration'].quantile(0.75))
    kernel_p90_dur = float(df_request_kernel['duration'].quantile(0.90))
    kernel_p95_dur = float(df_request_kernel['duration'].quantile(0.95))
    kernel_p99_dur = float(df_request_kernel['duration'].quantile(0.99))
    kernel_max_dur = float(df_request_kernel['duration'].max())
    kernel_var_dur = float(df_request_kernel['duration'].var())
    e2e_kernel_dur = float(sum(df_request_kernel['duration']))
    e2e_dur = float(e2e_kernel_dur + image_load_time)
    num_kerns = len(df_request_kernel['duration'])

    # print("kernel_min_dur:", kernel_min_dur)
    # print("kernel_p10_dur:", kernel_p10_dur)
    # print("kernel_p25_dur:", kernel_p25_dur)
    # print("kernel_mean_dur:", kernel_mean_dur)
    # print("kernel_median_dur:", kernel_median_dur)
    # print("kernel_p75_dur:", kernel_p75_dur)
    # print("kernel_p90_dur:", kernel_p90_dur)
    # print("kernel_p95_dur:", kernel_p95_dur)
    # print("kernel_p99_dur:", kernel_p99_dur)
    # print("kernel_max_dur:", kernel_max_dur)
    # print("kernel_var_dur:", kernel_var_dur)
    # print("e2e_kernel_dur:", e2e_kernel_dur)
    # print("e2e_dur:", e2e_dur)
    # print("num_kerns:", num_kerns)

    # Add up top 1, 2, 5, 7, 10, 20, 30, 40, 50 of kernel durations
    df_request_kernel = df_request_kernel.sort_values('duration', ascending=False)
    kernel_max_1 = df_request_kernel.head(1)['duration'].sum()
    kernel_max_2 = df_request_kernel.head(2)['duration'].sum()
    kernel_max_5 = df_request_kernel.head(5)['duration'].sum()
    kernel_max_7 = df_request_kernel.head(7)['duration'].sum()
    kernel_max_10 = df_request_kernel.head(10)['duration'].sum()
    kernel_max_20 = df_request_kernel.head(20)['duration'].sum()
    kernel_max_30 = df_request_kernel.head(30)['duration'].sum()
    kernel_max_40 = df_request_kernel.head(40)['duration'].sum()
    kernel_max_50 = df_request_kernel.head(50)['duration'].sum()

    kernel_max_list = []
    kernel_max_list.append(kernel_max_1)
    kernel_max_list.append(kernel_max_2)
    kernel_max_list.append(kernel_max_5)
    kernel_max_list.append(kernel_max_7)
    kernel_max_list.append(kernel_max_10)
    kernel_max_list.append(kernel_max_20)
    kernel_max_list.append(kernel_max_30)
    kernel_max_list.append(kernel_max_40)
    kernel_max_list.append(kernel_max_50)

    return load_time, image_load_time, model_size

        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vgg11', 
                            help='model name to extract profile data for (e.g., vgg11)')
    parser.add_argument('--batch_size', type=str, default=16,
                            help='batch size for the model to extract profile data fro (e.g., 16)')                            
    parser.add_argument('--profile_dir', type=str, default='./data/a100/vgg11/',
                            help='path to directory containing the profiling files')
    args = parser.parse_args()
    
    process_raw_ncu_data(profile_dir=args.profile_dir, model=args.model, batch_size=args.batch_size)
    df = aggregate_raw_ncu_data(profile_dir=args.profile_dir, model=args.model, batch_size=args.batch_size)
    model_load_time, image_load_time, model_size = extract_nsys_data(profile_dir=args.profile_dir, model=args.model, batch_size=args.batch_size)
    df['model_load_time'] = model_load_time
    df['image_load_time'] = image_load_time
    df['model_size'] = model_size
    df['model'] = args.model
    df['batch_size'] = args.batch_size
    first_col = df.pop('model')
    df.insert(0, 'model', first_col)
    second_col = df.pop('batch_size')
    df.insert(1, 'batch_size', second_col)

    profiler_database_path = './data/a100/model_profiles.csv'
    if os.path.exists(profiler_database_path):
        df.to_csv(profiler_database_path, mode='a', header=False, index=False)
    else:
        df.to_csv(profiler_database_path, index=False)
    