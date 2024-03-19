import os
import grpc
import json
import sys
import threading
from concurrent import futures
import subprocess
import time
import argparse
import json

sys.path.insert(0, '../generated')
import tb_controller_pb2
import tb_controller_pb2_grpc

def create_status(enum_status, msg):
    status = {}
    status['status'] = enum_status
    status['msg'] = msg
    return status

class TieBreaker_Controller(tb_controller_pb2_grpc.TieBreaker_ControllerServicer):
    def __init__(self):
        self.supported_models = []
        self.supported_batchs = [2, 4, 8, 16, 32]
        self.max_supported_size = 4
        self.job_mix_deployment_params = {} # job mix -> [distro type, rps, slo percentile, job_start_time, pid, gpu device id, high load mechanism]
        self.device_status = {}

        # Parse server config file
        with open(args.device_config, 'r') as file:
            data = json.load(file)
            for entry in data['server']:
                self.device_status[entry['device_id']] = [entry['value'], 'AVAILABLE']
        self.no_server_devices = len(self.device_status)

        print(self.device_status)
        print(self.no_server_devices)

        # Thread to monitor how long a job mix has been running for
        self.monitor_thread = threading.Thread(target=self.monitor_job_params)
        self.monitor_thread.daemon=True # # Daemonize the thread so it terminates when the main thread exits
        self.monitor_thread.start()
    
    def monitor_job_params(self):
        exp_duration = args.exp_duration
        while True:
            current_time = time.time()
            for job_mix, params in list(self.job_mix_deployment_params.items()):
                job_start_time = params[3]
                if current_time >= job_start_time + exp_duration:
                    echo_dict = {"mode": "stop", "device-id": params[5]}
                    echo_dict_string = json.dumps(echo_dict)
                    echo_cmd = "echo \'" + echo_dict_string + f"\' > /tmp/{params[4]}"
                    print('Echo cmd is ' + echo_cmd)
                    echo_tmp = subprocess.Popen(echo_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
                    self.device_status[params[5]] = 'AVAILABLE'
                    del self.job_mix_deployment_params[job_mix]
            time.sleep(0.5)
    
    def DeployJobMix(self, request, context):

        # TODO: CHECK MODEL TYPES, MODELS, BATCHES, AND JOB SIZES SUPPORTED
        
        # Assuming even number of A100s, for every pair one MPS and one MIG
        if len(self.job_mix_deployment_params) == (self.no_server_devices / 2):
            return tb_controller_pb2.DeploymentResponse(status=create_status('FAILURE', f'Server GPUs are occupied currently.'))

        # Construct string payload
        payloads = []
        for i, model in enumerate(request.models):
            payloads.append(
                {
                    "model-type": request.model_types[i],
                    "model": model,
                    "batch-size": request.batch_sizes[i],
                    "distribution-type": request.distro_type,
                    "rps": request.rps,
                    "slo-percentile": request.slo_percentile,
                    "slo-lw": request.slos[i],
                    "slo-hw": request.slos[i],
                    "ctrl-grpc": args.port
                }
            )

        # Launch models
        payload_string = json.dumps(payloads)
        deploy_cmd = "~/llama2-expr/run_job_mix.sh --device-type a100 --tie-breaker --load 1 \'" + payload_string + '\''
        print('Deployment command is: ' + deploy_cmd)
        deploy_tmp = subprocess.Popen(deploy_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
        pid = deploy_tmp.pid
        print(f'PID is {pid}')
        # pid = -1

        # Find first available MPS device to launch model on
        mps_device = -1
        for device, value in list(self.device_status.items()):
            if (value[0] == 'MPS') and (value[1] == 'AVAILABLE'):
                updated_value = value
                updated_value[1] = 'OCCUPIED'
                self.device_status[device] = updated_value
                mps_device = device
                break
        
        # Didn't find available MPS device, shouldn't come here ever
        if mps_device == -1:
            return tb_controller_pb2.DeploymentResponse(status=create_status('FAILURE', f'Server GPUs are occupied currently. Should not end up here!'))

        # Echo mode and device to launch models on
        echo_dict = {"mode": "mps-uncap", "device-id": mps_device}
        echo_dict_string = json.dumps(echo_dict)
        echo_cmd = "echo \'" + echo_dict_string + f"\' > /tmp/{pid}"
        print('Echo cmd is ' + echo_cmd)
        while not os.path.exists(f'/tmp/{pid}'):
            time.sleep(0.5)
        echo_tmp = subprocess.Popen(echo_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
        job_mix_start_time = time.time()

        # Store job mix state
        job_mix_string = ""
        for i, model in enumerate(request.models):
            if job_mix_string == "":
                job_mix_string = model + '-' + str(request.batch_sizes[i]) + '-' + str(request.slos[i])
            else:
                job_mix_string += " " + model + '-' + str(request.batch_sizes[i]) + '-' + str(request.slos[i])

        # Consult TieBreaker model for concurrency mechanism to use during high load
        # TODO: consult mechanism here
        high_load_mechanism = 'mig'

        self.job_mix_deployment_params[job_mix_string] = [request.distro_type, request.rps, request.slo_percentile, job_mix_start_time, pid, mps_device, high_load_mechanism]
        print()
        print(job_mix_string)
        print(self.job_mix_deployment_params)
        return tb_controller_pb2.DeploymentResponse(status=create_status('SUCCESS', f'Succesfully deployed models on an MPS GPU!'))
    
    def MigrateJobMix(self, request, context):
        for job_mix, params in list(self.job_mix_deployment_params.items()):
            # Need to live migrate
            if (params[5] == request.gpu_no) and (params[-1] == 'mig'):

                # Find available MIG GPU
                mig_device = -1
                for device, value in list(self.device_status.items()):
                    if (value[0] == 'MIG') and (value[1] == 'AVAILABLE'):
                        updated_value = value
                        updated_value[1] = 'OCCUPIED'
                        self.device_status[device] = updated_value
                        mig_device = device
                        break
                
                # Failed to find available MIG GPU
                if mig_device == -1:
                    return tb_controller_pb2.MigrationResponse(status=create_status('FAILURE', f'MIG GPUs unavailable!'))
                
                # Issue echo command to kickstart live migration
                echo_dict = {"mode": f"{params[-1]}", "device-id": mig_device}
                echo_dict_string = json.dumps(echo_dict)
                echo_cmd = "echo \'" + echo_dict_string + f"\' > /tmp/{params[4]}"
                print('Echo cmd is ' + echo_cmd)
                echo_tmp = subprocess.Popen(echo_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)

                # Update state -- old device is available
                update_device_value = self.device_status[params[5]]
                update_device_value[1] = 'AVAILABLE'
                self.device_status[params[5]] = update_device_value

                # Update job mix state -- specify new device
                updated_params = params
                updated_params[5] = mig_device
                self.job_mix_deployment_params[job_mix] = updated_params
            # No need for live migratino
            else:
                return tb_controller_pb2.MigrationResponse(status=create_status('SUCCESS', f'No need to live migrate the job mix per TieBreaker Model!'))


        return tb_controller_pb2.MigrationResponse(status=create_status('FAILURE', f'Could not find job mix with given request gpu no {request.gpu_no}'))

def tiebreaker_controller():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=40))
    tb_controller_pb2_grpc.add_TieBreaker_ControllerServicer_to_server(TieBreaker_Controller(), server)
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()
    print('TieBreaker Controller is up and running!')
    server.wait_for_termination()

if __name__=='__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--exp-duration", type=int, default=60)
    parser.add_argument("--port", type=str, default=50053)
    parser.add_argument("--device-config", type=str, default='../device-config.json')
    args = parser.parse_args()
    tiebreaker_controller()