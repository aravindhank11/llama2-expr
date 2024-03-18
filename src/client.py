import grpc
import sys

sys.path.insert(0, '../generated')
import tb_controller_pb2
import tb_controller_pb2_grpc

with grpc.insecure_channel('localhost:50053') as channel:
    stub = tb_controller_pb2_grpc.TieBreaker_ControllerStub(channel)
    response = stub.DeployJobMix(tb_controller_pb2.DeploymentRequest(
        model_types = ['vision', 'vision', 'vision'],
        models = ['vgg19', 'mobilenet_v2', 'vgg11'],
        batch_sizes = [2, 4, 8],
        slos = [1000, 1000, 1000],
        distro_type = 'closed',
        rps = 100000,
        slo_percentile = 90
    ))
    print(response.status)