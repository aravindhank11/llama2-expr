import grpc
import sys
import time

sys.path.insert(0, '../generated')
import tb_controller_pb2
import tb_controller_pb2_grpc

with grpc.insecure_channel('localhost:50053') as channel:
    stub = tb_controller_pb2_grpc.TieBreaker_ControllerStub(channel)
    deployment_response = stub.DeployJobMix(tb_controller_pb2.DeploymentRequest(
        model_types = ['vision', 'vision'],
        models = ['vgg19', 'mobilenet_v2'],
        batch_sizes = [2, 4],
        slos = [1000, 1000],
        distro_type = 'closed',
        rps = 100000,
        slo_percentile = 90
    ))
    print(deployment_response.status)

    time.sleep(20)

    migration_response = stub.MigrateJobMix(tb_controller_pb2.MigrationRequest(
        unique_mix_id = 1,
        breached_status = 'high_wm'
    ))
    print(migration_response.status)
