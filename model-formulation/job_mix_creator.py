from random import choices, shuffle
import sqlite3
import ast

MODEL_CONFIGS_250 = [['mobilenet_v2', 32, 23450.0], ['densenet121', 32, 59535.0], ['densenet201', 32, 94150.0], ['vgg19', 32, 77805.0], ['inception_v3', 32, 39270.0], ['resnet50', 32, 46585.0], ['resnet18', 32, 17955.0], ['resnet34', 32, 25375.0], ['resnet101', 32, 70035.0], ['resnet152', 32, 95725.0], ['densenet161', 32, 111125.0], ['densenet169', 32, 75495.0], ['vgg11', 32, 39620.0], ['vgg13', 32, 57400.0], ['vgg16', 32, 67620.0], ['alexnet', 32, 10920.0], ['squeezenet1_0', 32, 22785.0], ['squeezenet1_1', 32, 15995.0], ['mobilenet_v3_large', 32, 25025.0], ['mobilenet_v3_small', 32, 20230.0], ['efficientnet_b5', 32, 106645.0], ['efficientnet_b6', 32, 140455.0], ['efficientnet_b7', 32, 187285.0], ['mobilenet_v2', 2, 20650.0], ['densenet121', 2, 55545.0], ['densenet201', 2, 95025.0], ['vgg19', 2, 8855.0], ['inception_v3', 2, 40180.0], ['resnet50', 2, 24115.0], ['resnet18', 2, 8330.0], ['resnet34', 2, 13825.0], ['resnet101', 2, 43785.0], ['resnet152', 2, 63560.0], ['densenet161', 2, 78120.0], ['densenet169', 2, 78890.0], ['vgg11', 2, 5530.0], ['vgg13', 2, 6755.0], ['vgg16', 2, 7770.0], ['alexnet', 2, 3570.0], ['squeezenet1_0', 2, 9485.0], ['squeezenet1_1', 2, 8680.0], ['mobilenet_v3_large', 2, 24395.0], ['mobilenet_v3_small', 2, 20895.0], ['efficientnet_b5', 2, 74445.0], ['efficientnet_b6', 2, 85575.0], ['efficientnet_b7', 2, 102865.0], ['mobilenet_v2', 4, 20370.0], ['densenet121', 4, 55545.0], ['densenet201', 4, 95200.0], ['vgg19', 4, 13020.0], ['inception_v3', 4, 40425.0], ['resnet50', 4, 23625.0], ['resnet18', 4, 8295.0], ['resnet34', 4, 13720.0], ['resnet101', 4, 43785.0], ['resnet152', 4, 64085.0], ['densenet161', 4, 77210.0], ['densenet169', 4, 78855.0], ['vgg11', 4, 7560.0], ['vgg13', 4, 10010.0], ['vgg16', 4, 11515.0], ['alexnet', 4, 3570.0], ['squeezenet1_0', 4, 9590.0], ['squeezenet1_1', 4, 8820.0], ['mobilenet_v3_large', 4, 24605.0], ['mobilenet_v3_small', 4, 20755.0], ['efficientnet_b5', 4, 73149.99999999999], ['efficientnet_b6', 4, 85855.0], ['efficientnet_b7', 4, 107660.0], ['mobilenet_v2', 8, 20475.0], ['densenet121', 8, 55685.0], ['densenet201', 8, 95760.0], ['vgg19', 8, 23275.0], ['inception_v3', 8, 40880.0], ['resnet50', 8, 23975.0], ['resnet18', 8, 8960.0], ['resnet34', 8, 14035.0], ['resnet101', 8, 44135.0], ['resnet152', 8, 64155.0], ['densenet161', 8, 77105.0], ['densenet169', 8, 78680.0], ['vgg11', 8, 13055.0], ['vgg13', 8, 17815.0], ['vgg16', 8, 20475.0], ['alexnet', 8, 4585.0], ['squeezenet1_0', 8, 9765.0], ['squeezenet1_1', 8, 8855.0], ['mobilenet_v3_large', 8, 24710.0], ['mobilenet_v3_small', 8, 21385.0], ['efficientnet_b5', 8, 74760.0], ['efficientnet_b6', 8, 87815.0], ['efficientnet_b7', 8, 107310.0], ['mobilenet_v2', 16, 20440.0], ['densenet121', 16, 56980.0], ['densenet201', 16, 93695.0], ['vgg19', 16, 41370.0], ['inception_v3', 16, 40355.0], ['resnet50', 16, 26390.0], ['resnet18', 16, 10535.0], ['resnet34', 16, 15085.0], ['resnet101', 16, 43995.0], ['resnet152', 16, 64330.0], ['densenet161', 16, 76965.0], ['densenet169', 16, 78470.0], ['vgg11', 16, 21770.0], ['vgg13', 16, 30835.0], ['vgg16', 16, 36050.0], ['alexnet', 16, 6755.0], ['squeezenet1_0', 16, 13020.0], ['squeezenet1_1', 16, 9660.0], ['mobilenet_v3_large', 16, 25585.0], ['mobilenet_v3_small', 16, 21490.0], ['efficientnet_b5', 16, 78610.0], ['efficientnet_b6', 16, 90055.0], ['efficientnet_b7', 16, 110390.0]]
DATA_STORE = DATA_STORE = '/home/ps35324/gpu-sharing-scheduler/src/data-store/scheduler.db'

def determine_capapble_mig():
    conn = sqlite3.connect(DATA_STORE)
    cur = conn.cursor()

    col_to_mig = {'exec_7g': '7g.40gb', 'exec_4g': '4g.20gb', 'exec_3g': '3g.20gb', 'exec_2g': '2g.10gb', 'exec_1g': '1g.10gb'}

    slo_mig_combos = {}

    # Get MIG share that will meet 3.5x for each model
    slo_250 = {}
    for model_name, batch_size, slo in MODEL_CONFIGS_250:
        cur.execute(f"SELECT exec_7g, exec_4g, exec_3g, exec_2g, exec_1g FROM model_profiles WHERE model_name = ? AND max_batch_size = ? AND model_id > 18 LIMIT 1", (model_name, batch_size))
        row = cur.fetchone()

        if row is None:
            print(f"No row found for model '{model_name}'")
        else:
            exec_columns = ['exec_7g', 'exec_4g', 'exec_3g', 'exec_2g', 'exec_1g']
            valid_columns = [column for column in exec_columns if row[exec_columns.index(column)] < slo]

            if valid_columns:
                selected_column = min(valid_columns, key=lambda x: int(x[5]))
                slo_250[model_name + '_' + str(batch_size)] = col_to_mig[selected_column]
            else:
                print(f"For model '{model_name}': No column with a value less than SLO {slo} for slo 1.05x")
    slo_mig_combos['MODEL_CONFIGS_250'] = slo_250

    conn.close()
    return slo_mig_combos

def create_valid_combo_set(combo_list):
    capable_migs = determine_capapble_mig()
    valid_combos = []
    for combo in combo_list:
        slices = []
        new_combo = []
        for model in combo:
            model_breakdown = model.split('-')
            share = int(capable_migs['MODEL_CONFIGS_250'][model_breakdown[0] + '_' + str(model_breakdown[1])][0])
            slices.append(share)
            new_model = [model_breakdown[0], int(model_breakdown[1]), float(model_breakdown[2]), share]
            new_combo.append(new_model)
        
        if sum(slices) <= 7:
            valid_combos.append(new_combo)
    return valid_combos

def find_min_indices(list):
    return_list = []
    for index, value in enumerate(list):
        if value == min(list):
            return_list.append(index)
    return return_list

def find_max_indices(list):
    return_list = []
    for index, value in enumerate(list):
        if value == max(list):
            return_list.append(index)
    return return_list

def assign_mig_slice(combo_list):
    reassigned_list = []
    for combo in combo_list:
        # print(combo)
        # Check if mig assignments already equal 7, if so no need to reassign
        if sum([model[3] for model in combo]) < 7:
            # 2 models co-locatd: assign 4 and 3 if not already up to 7
            if len(combo) == 2:
                max_index = find_max_indices([model[3] for model in combo])
                max_index = find_max_indices([model[1] if i in max_index else -1 for i, model in enumerate(combo)])
                for i, model in enumerate(combo):
                    if i == max_index[0]:
                        combo[i][3] = 4
                    else:
                        combo[i][3] = 3
            
            # 3 models co-located: assign 3, 2, 2 if not already up to 7
            elif len(combo) == 3:
                max_index = find_max_indices([model[3] for model in combo])
                max_index = find_max_indices([model[1] if i in max_index else -1 for i, model in enumerate(combo)])
                for i, model in enumerate(combo):
                    if i == max_index[0]:
                        combo[i][3] = 3
                    else:
                        combo[i][3] = 2

            # 4 models co-located: assign 2, 2, 2, 1 if not already up to 7
            elif len(combo) == 4:
                min_index = find_min_indices([model[3] for model in combo])
                min_index = find_min_indices([model[1] if i in min_index else 1000000 for i, model in enumerate(combo)])
                for i, model in enumerate(combo):
                    if i == min_index[0]:
                        combo[i][3] = 1
                    else:
                        combo[i][3] = 2

            # 5 models co-located: assign 2, 2, 1, 1, 1 if not already up to 7
            elif len(combo) == 5:
                # Do this logic once to get one max
                max_index = find_max_indices([model[3] for model in combo])
                max_index = find_max_indices([model[1] if i in max_index else -1 for i, model in enumerate(combo)])
                # Do this logic again to get another max, slightly tweaked
                max_index2 = find_max_indices([-1 if i == max_index[0] else model[3] for i, model in enumerate(combo)])
                max_index2 = find_max_indices([model[1] if i in max_index2 else -1 for i, model in enumerate(combo)])
                for i, model in enumerate(combo):
                    if (i == max_index[0]) or (i == max_index2[0]):
                        combo[i][3] = 2
                    else:
                        combo[i][3] = 1

            # 6 models co-located: 2, 1, 1, 1, 1, 1 if not already up to 7
            elif len(combo) == 6:
                max_index = find_max_indices([model[3] for model in combo])
                max_index = find_max_indices([model[1] if i in max_index else -1 for i, model in enumerate(combo)])
                for i, model in enumerate(combo):
                    if i == max_index[0]:
                        combo[i][3] = 2
                    else:
                        combo[i][3] = 1
                
            # 7 models co-located: assign all of them a slice of 1 if not already up to 7
            elif len(combo) == 7:
                for model in combo:
                    if (model[3] != 1):
                        print(combo)
        # else:
        #     print(combo)
        reassigned_list.append(combo)

    return reassigned_list

def create_valid_combos(num_models):

    combos = []
    if num_models == 3:
        # Create list of all possible combinatinos of the models for the given co-location number
        for j1 in MODEL_CONFIGS_250:
            for j2 in MODEL_CONFIGS_250:
                for j3 in MODEL_CONFIGS_250:
                    combos.append([j1, j2, j3])
    elif num_models == 2:
        # Create list of all possible combinatinos of the models for the given co-location number
        for j1 in MODEL_CONFIGS_250:
            for j2 in MODEL_CONFIGS_250:
                combos.append([j1, j2])
    elif num_models == 4:
        # Create list of all possible combinatinos of the models for the given co-location number
        for j1 in MODEL_CONFIGS_250:
            for j2 in MODEL_CONFIGS_250:
                for j3 in MODEL_CONFIGS_250:
                    for j4 in MODEL_CONFIGS_250:
                        combos.append([j1, j2, j3, j4])
                        if len(combos) == 200000:
                            break 
                    if len(combos) == 200000:
                            break 
                if len(combos) == 200000:
                            break 
            if len(combos) == 200000:
                            break 
        
    print(len(combos))

    # Remove duplicates -- only keep unique combinations
    unique_combos = set()
    for combo in combos:
        string_list = []
        for model in combo:
            string_list.append(model[0] + '-' + str(model[1]) + '-' + str(model[2]))
        unique_combos.add(tuple(sorted(string_list)))
    unique_combos = list([list(tuple_combo) for tuple_combo in unique_combos])

    # Create valid combos only those that can guarantee fit on MIG
    valid_combos = assign_mig_slice(create_valid_combo_set(unique_combos))

    return valid_combos

def filter_tested_combos(num_models):
    # Get combinations tested already
    combo_file = open(f'/home/ps35324/gpu-sharing-scheduler/src/master/combos_{num_models}.txt')
    data = combo_file.read()
    combos = data.split('\n')
    combo_file.close()
    tested_combos = []
    for combo in combos:
        input_string_with_comma = combo.replace("] [", "], [")
        result_lists = ast.literal_eval("[" + input_string_with_comma + "]")
        if len(result_lists) > 0:
            tested_combos.append(result_lists)

    # Filter out combinations already tested
    tested_combos_set = set()
    for combo in tested_combos:
        string_list = []
        for model in combo:
            string_list.append(model[0] + '-' + str(model[1]) + '-' + str(model[2]) + '-' + str(model[3]))
        tested_combos_set.add(tuple(sorted(string_list)))
    print(len(tested_combos_set))
    valid_combos = create_valid_combos(num_models)
    valid_combos_set = set()
    for combo in valid_combos:
        string_list = []
        for model in combo:
            string_list.append(model[0] + '-' + str(model[1]) + '-' + str(model[2]) + '-' + str(model[3]))
        valid_combos_set.add(tuple(sorted(string_list)))
    print(len(valid_combos_set))

    filtered_combos_set = valid_combos_set - tested_combos_set
    filtered_combos = list([list(tuple_combo) for tuple_combo in filtered_combos_set])
    shuffle(filtered_combos)

    # Save the combinations to a file -- this is done because we can't reproduce the same combination orders do to sets being used
    with open('./job_mixes/combos_{}.txt'.format(num_models), 'w') as fp:
        for sublist in filtered_combos:
            if sublist is not None:    # filter out None items
                print(*sublist, file=fp)

if __name__=='__main__':
    # filter_tested_combos(3)
    filter_tested_combos(4)



