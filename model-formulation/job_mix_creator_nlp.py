import random

MODEL_CONFIGS_250 = [['mobilenet_v2', 32, 23450.0], ['densenet121', 32, 59535.0], ['densenet201', 32, 94150.0], ['vgg19', 32, 77805.0], ['inception_v3', 32, 39270.0], ['resnet50', 32, 46585.0], ['resnet18', 32, 17955.0], ['resnet34', 32, 25375.0], ['resnet101', 32, 70035.0], ['resnet152', 32, 95725.0], ['densenet161', 32, 111125.0], ['densenet169', 32, 75495.0], ['vgg11', 32, 39620.0], ['vgg13', 32, 57400.0], ['vgg16', 32, 67620.0], ['alexnet', 32, 10920.0], ['squeezenet1_0', 32, 22785.0], ['squeezenet1_1', 32, 15995.0], ['mobilenet_v3_large', 32, 25025.0], ['mobilenet_v3_small', 32, 20230.0], ['efficientnet_b5', 32, 106645.0], ['efficientnet_b6', 32, 140455.0], ['efficientnet_b7', 32, 187285.0], ['mobilenet_v2', 2, 20650.0], ['densenet121', 2, 55545.0], ['densenet201', 2, 95025.0], ['vgg19', 2, 8855.0], ['inception_v3', 2, 40180.0], ['resnet50', 2, 24115.0], ['resnet18', 2, 8330.0], ['resnet34', 2, 13825.0], ['resnet101', 2, 43785.0], ['resnet152', 2, 63560.0], ['densenet161', 2, 78120.0], ['densenet169', 2, 78890.0], ['vgg11', 2, 5530.0], ['vgg13', 2, 6755.0], ['vgg16', 2, 7770.0], ['alexnet', 2, 3570.0], ['squeezenet1_0', 2, 9485.0], ['squeezenet1_1', 2, 8680.0], ['mobilenet_v3_large', 2, 24395.0], ['mobilenet_v3_small', 2, 20895.0], ['efficientnet_b5', 2, 74445.0], ['efficientnet_b6', 2, 85575.0], ['efficientnet_b7', 2, 102865.0], ['mobilenet_v2', 4, 20370.0], ['densenet121', 4, 55545.0], ['densenet201', 4, 95200.0], ['vgg19', 4, 13020.0], ['inception_v3', 4, 40425.0], ['resnet50', 4, 23625.0], ['resnet18', 4, 8295.0], ['resnet34', 4, 13720.0], ['resnet101', 4, 43785.0], ['resnet152', 4, 64085.0], ['densenet161', 4, 77210.0], ['densenet169', 4, 78855.0], ['vgg11', 4, 7560.0], ['vgg13', 4, 10010.0], ['vgg16', 4, 11515.0], ['alexnet', 4, 3570.0], ['squeezenet1_0', 4, 9590.0], ['squeezenet1_1', 4, 8820.0], ['mobilenet_v3_large', 4, 24605.0], ['mobilenet_v3_small', 4, 20755.0], ['efficientnet_b5', 4, 73149.99999999999], ['efficientnet_b6', 4, 85855.0], ['efficientnet_b7', 4, 107660.0], ['mobilenet_v2', 8, 20475.0], ['densenet121', 8, 55685.0], ['densenet201', 8, 95760.0], ['vgg19', 8, 23275.0], ['inception_v3', 8, 40880.0], ['resnet50', 8, 23975.0], ['resnet18', 8, 8960.0], ['resnet34', 8, 14035.0], ['resnet101', 8, 44135.0], ['resnet152', 8, 64155.0], ['densenet161', 8, 77105.0], ['densenet169', 8, 78680.0], ['vgg11', 8, 13055.0], ['vgg13', 8, 17815.0], ['vgg16', 8, 20475.0], ['alexnet', 8, 4585.0], ['squeezenet1_0', 8, 9765.0], ['squeezenet1_1', 8, 8855.0], ['mobilenet_v3_large', 8, 24710.0], ['mobilenet_v3_small', 8, 21385.0], ['efficientnet_b5', 8, 74760.0], ['efficientnet_b6', 8, 87815.0], ['efficientnet_b7', 8, 107310.0], ['mobilenet_v2', 16, 20440.0], ['densenet121', 16, 56980.0], ['densenet201', 16, 93695.0], ['vgg19', 16, 41370.0], ['inception_v3', 16, 40355.0], ['resnet50', 16, 26390.0], ['resnet18', 16, 10535.0], ['resnet34', 16, 15085.0], ['resnet101', 16, 43995.0], ['resnet152', 16, 64330.0], ['densenet161', 16, 76965.0], ['densenet169', 16, 78470.0], ['vgg11', 16, 21770.0], ['vgg13', 16, 30835.0], ['vgg16', 16, 36050.0], ['alexnet', 16, 6755.0], ['squeezenet1_0', 16, 13020.0], ['squeezenet1_1', 16, 9660.0], ['mobilenet_v3_large', 16, 25585.0], ['mobilenet_v3_small', 16, 21490.0], ['efficientnet_b5', 16, 78610.0], ['efficientnet_b6', 16, 90055.0], ['efficientnet_b7', 16, 110390.0]]
DATA_STORE = DATA_STORE = '/home/ps35324/gpu-sharing-scheduler/src/data-store/scheduler.db'


def create_job_mix_size_2():
    batches = [2, 4, 8, 16, 32]
    nlp_models = ['bert', 'llama']
    size_2_jobs = []
    # Create 10 job mixes with a bert and vision job
    for i in range(10):
        bert_str = f'bert-bert-{random.choice(batches)}'
        vision_job = MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str = f'vision-{vision_job[0]}-{vision_job[1]}'
        size_2_jobs.append([f'{bert_str} {vision_str}'])
    # Create 10 job mixes with a llama and vision job
    for i in range(10):
        llama_str = f'llama-llama-{random.choice(batches)}'
        vision_job = MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str = f'vision-{vision_job[0]}-{vision_job[1]}'
        size_2_jobs.append([f'{llama_str} {vision_str}'])
    # Create 10 job mixes with a bert and llama job
    for i in range(10):
        bert_str = f'bert-bert-{random.choice(batches)}'
        llama_str = f'llama-llama-{random.choice(batches)}'
        size_2_jobs.append([f'{llama_str} {bert_str}'])

    # Save the combinations to a file
    with open('./job_mixes/combos_2_nlp.txt', 'w') as fp:
        for string in size_2_jobs:
            print(*string, file=fp)

def create_job_mix_size_3():
    batches = [2, 4, 8, 16, 32]
    nlp_models = ['bert', 'llama']
    size_3_jobs = []
    # Create 10 job mixes with a bert and two vision jobs
    for i in range(10):
        bert_str = f'bert-bert-{random.choice(batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        vision_job2= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str2 = f'vision-{vision_job2[0]}-{vision_job2[1]}'
        size_3_jobs.append([f'{bert_str} {vision_str1} {vision_str2}'])
    # Create 10 job mixes with a llama and two vision jobs
    for i in range(10):
        llama_str = f'llama-llama-{random.choice(batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        vision_job2= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str2 = f'vision-{vision_job2[0]}-{vision_job2[1]}'
        size_3_jobs.append([f'{llama_str} {vision_str1} {vision_str2}'])
    # Create 10 job mixes with a bert, llama, and vision job
    for i in range(10):
        bert_str = f'bert-bert-{random.choice(batches)}'
        llama_str = f'llama-llama-{random.choice(batches)}'
        vision_job = MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str = f'vision-{vision_job[0]}-{vision_job[1]}'
        size_3_jobs.append([f'{llama_str} {bert_str} {vision_str}'])

    # Save the combinations to a file
    with open('./job_mixes/combos_3_nlp.txt', 'w') as fp:
        for string in size_3_jobs:
            print(*string, file=fp)

def create_job_mix_size_4():
    bert_batches = [2, 4, 8, 16, 32]
    batches = [2, 4, 8, 16]
    size_4_jobs_list = []
    # Create 5 job mixes with 1 bert and 3 vision models
    size_4_jobs_set = set()
    while True:
        bert_str = f'bert-bert-{random.choice(bert_batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        vision_job2= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str2 = f'vision-{vision_job2[0]}-{vision_job2[1]}'
        vision_job3= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str3 = f'vision-{vision_job3[0]}-{vision_job3[1]}'
        job_mix_str = f'{bert_str} {vision_str1} {vision_str2} {vision_str3}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break

    # Create 5 job mixes with 1 llama and 3 vision models
    size_4_jobs_set = set()
    while True:
        llama_str = f'llama-llama-{random.choice(batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        vision_job2= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str2 = f'vision-{vision_job2[0]}-{vision_job2[1]}'
        vision_job3= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str3 = f'vision-{vision_job3[0]}-{vision_job3[1]}'
        job_mix_str = f'{llama_str} {vision_str1} {vision_str2} {vision_str3}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break
    # Create 5 job mixes with 1 llama, 1 bert, and 2 vision models
    size_4_jobs_set = set()
    while True:
        llama_str = f'llama-llama-{random.choice(batches)}'
        bert_str = f'bert-bert-{random.choice(bert_batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        vision_job2= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str2 = f'vision-{vision_job2[0]}-{vision_job2[1]}'
        job_mix_str = f'{llama_str} {bert_str} {vision_str1} {vision_str2}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break

    # Create 5 job mixes with 2 bert and 2 vision models
    size_4_jobs_set = set()
    while True:
        bert_str1 = f'bert-bert-{random.choice(bert_batches)}'
        bert_str2 = f'bert-bert-{random.choice(bert_batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        vision_job2= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str2 = f'vision-{vision_job2[0]}-{vision_job2[1]}'
        job_mix_str = f'{bert_str1} {bert_str2} {vision_str1} {vision_str2}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break

    # Create 5 job mixes with 2 llama and 2 vision models
    size_4_jobs_set = set()
    while True:
        llama_str1 = f'llama-llama-{random.choice(batches)}'
        llama_str2 = f'llama-llama-{random.choice(batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        vision_job2= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str2 = f'vision-{vision_job2[0]}-{vision_job2[1]}'
        job_mix_str = f'{llama_str1} {llama_str2} {vision_str1} {vision_str2}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break

    # Create 5 job mixes with 3 bert and 1 vision model
    size_4_jobs_set = set()
    while True:
        bert_str1 = f'bert-bert-{random.choice(bert_batches)}'
        bert_str2 = f'bert-bert-{random.choice(bert_batches)}'
        bert_str3 = f'bert-bert-{random.choice(bert_batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        job_mix_str = f'{bert_str1} {bert_str2} {bert_str3} {vision_str1}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break

    # Create 5 job mixes with 3 llama and 1 vision model
    size_4_jobs_set = set()
    while True:
        llama_str1 = f'llama-llama-{random.choice(batches)}'
        llama_str2 = f'llama-llama-{random.choice(batches)}'
        llama_str3 = f'llama-llama-{random.choice(batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        job_mix_str = f'{llama_str1} {llama_str2} {llama_str3} {vision_str1}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break

    # Create 5 job mixes with 2 bert, 1 llama, 1 vision model
    size_4_jobs_set = set()
    while True:
        llama_str = f'llama-llama-{random.choice(batches)}'
        bert_str1 = f'bert-bert-{random.choice(bert_batches)}'
        bert_str2 = f'bert-bert-{random.choice(bert_batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        job_mix_str = f'{llama_str} {bert_str1} {bert_str2} {vision_str1}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break
    # Create 5 job mixes with 2 llama, 1 bert, 1 vision model
    size_4_jobs_set = set()
    while True:
        llama_str1 = f'llama-llama-{random.choice(batches)}'
        llama_str2 = f'llama-llama-{random.choice(batches)}'
        bert_str = f'bert-bert-{random.choice(bert_batches)}'
        vision_job1= MODEL_CONFIGS_250[random.randint(0, 114)]
        vision_str1 = f'vision-{vision_job1[0]}-{vision_job1[1]}'
        job_mix_str = f'{llama_str1} {llama_str2} {bert_str} {vision_str1}'
        set_init_size = len(size_4_jobs_set)
        size_4_jobs_set.add(' '.join(sorted(job_mix_str.split(' '))))
        set_insert_size = len(size_4_jobs_set)
        if set_insert_size > set_init_size:
            size_4_jobs_list.append([job_mix_str])
        if len(size_4_jobs_set) == 5:
            break
    
    # Save the combinations to a file
    with open('./job_mixes/combos_4_nlp.txt', 'w') as fp:
        for string in size_4_jobs_list:
            print(*string, file=fp)


if __name__=='__main__':
    # create_job_mix_size_2()
    # create_job_mix_size_3()
    create_job_mix_size_4()