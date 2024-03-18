import os
import grpc
import json
import sys
import threading
from concurrent import futures
import subprocess
import time
import argparse

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
        self.current_device = 0

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
                    while not os.path.exists(f'/tmp/{params[4]}'):
                        time.sleep(0.5)
                    echo_tmp = subprocess.Popen(echo_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
                    job_mix_start_time = time.time()
            time.sleep(0.5)
    
    def DeployJobMix(self, request, context):

        # TODO: CHECK MODEL TYPES, MODELS, BATCHES, AND JOB SIZES SUPPORTED

        if len(self.job_mix_deployment_params) > 2:
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
                    "ctrl-grpc": None
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

        # Echo mode and device to launch models on
        echo_dict = {"mode": "mps-uncap", "device-id": self.current_device}
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

        self.job_mix_deployment_params[job_mix_string] = [request.distro_type, request.rps, request.slo_percentile, job_mix_start_time, pid, self.current_device, high_load_mechanism]
        self.current_device += 1
        print()
        print(job_mix_string)
        print(self.job_mix_deployment_params)
        print(self.current_device)
        return tb_controller_pb2.DeploymentResponse(status=create_status('SUCCESS', f'Succesfully deployed models on an MPS GPU!'))
    
    def MigrateJobMix(self, request, context):
        
        

        # 1. Consult model --> should we migrate
        # 2. Create new process (run.sh again?)
        print('hello world')

def tiebreaker_controller():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=40))
    tb_controller_pb2_grpc.add_TieBreaker_ControllerServicer_to_server(TieBreaker_Controller(), server)
    server.add_insecure_port('[::]:50053')
    server.start()
    print('TieBreaker Controller is up and running!')
    server.wait_for_termination()

if __name__=='__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--exp-duration", type=int, default=60)
    args = parser.parse_args()
    tiebreaker_controller()