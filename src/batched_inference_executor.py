import os
import threading
import pickle
import argparse
import signal
import sys
import time
import timeit
import random
import torch
from ctypes import cdll
from queue import Queue
from queue import Empty as QueueEmptyException
import numpy as np
import grpc
from batched_inference import get_batched_inference_object
from utils import (
    OnlinePercentile,
    AtomicBoolean,
    DistributionType,
    WatermarkType,
    SysVQueue,
    log,
    orion_block,
)

# sys.path.insert(0, '../generated')
import tb_controller_pb2
import tb_controller_pb2_grpc


WARMUP_REQS = 10
LARGE_NUM_REQS = 100000
ORION_LIB = "/root/orion/src/cuda_capture/libinttemp.so"
SLO_HISTORY_BUFFER_SIZE = 1000


class BatchedInferenceExecutor:
    def __init__(
        self, model_obj, distribution_type, rps, tid,
        num_infer=sys.maxsize,
        thread_barrier=None,
        return_queue=None,
        duration=-1,
        qid=-1,
        slo_percentile=0.9,
        slo_lw=float("inf"),
        slo_hw=float("inf"),
        ctrl_grpc_channel=None,
    ):
        self.model_obj = model_obj
        self.num_infer = num_infer
        self.rps = rps
        self.qid = qid
        self.tid = tid
        self.duration = duration
        self.slo_low_wm = slo_lw
        self.slo_high_wm = slo_hw
        if slo_percentile < 0 or slo_percentile > 100:
            print("slo percentile must be float between 0 and 100.")
            print(f"Got: {slo_percentile}")
            sys.exit(1)
        self.opc = OnlinePercentile(slo_percentile, SLO_HISTORY_BUFFER_SIZE)
        self.ctrl_slo_comm_running = AtomicBoolean(False)

        self.ctrl_grpc_channel = ctrl_grpc_channel
        self.unique_mix_id = None
        self.stub = None
        if self.ctrl_grpc_channel:
            self._setup_ctrl_grpc_channel()

        # Thread synchronization mechanism
        self.thread_barrier = thread_barrier
        self.return_queue = return_queue

        # Process synchronization mechanism
        self.start = False
        self.finish = False
        self.job_completed = False

        self._orion_lib = None
        if os.path.exists(ORION_LIB):
            self._orion_lib = cdll.LoadLibrary(ORION_LIB)

        if distribution_type == "closed":
            self.distribution_type = DistributionType.CLOSED
        elif distribution_type == "point":
            self.distribution_type = DistributionType.POINT
            self._sleep_time = [1 / rps]
        elif distribution_type == "poisson":
            self.distribution_type = DistributionType.POISSON
            self._sleep_time = np.random.exponential(
                scale=(1 / rps),
                size=10000
            )
        else:
            print(f"Unknown distribution type: {distribution_type}")
            print("Allowed values: closed, point, poisson")
            sys.exit(1)

    def _setup_ctrl_grpc_channel(self):
        split_list = self.ctrl_grpc_channel.split(' ')
        if len(split_list) == 2:
            print("Creating stub to TieBreaker Controller")
            port = split_list[0]
            self.unique_mix_id = split_list[1]
            with grpc.insecure_channel(f'localhost:{port}'):
                self.stub = tb_controller_pb2_grpc.TieBreaker_ControllerStub(channel)
        
    def _convey_slo_breach(self, watermark_type):
        # watermark_type: HIGH => We violated high watermark
        #                 LOW  => We went below the low watermark
        print(f"Reporting SLO VIOLATION {watermark_type}")
        if (self.stub != None) and (self.unique_mix_id != None):
            migration_response = self.stub.MigrateJobMix(tb_controller_pb2.MigrationRequest(
                unique_mix_id=self.unique_mix_id,
                breached_status = watermark_type[0]
            ))
            print(f"Migration response: {migration_response.status}")

        # Once communicated flip the atomic variable
        self.ctrl_slo_comm_running.flip()

    def _slo_check(self):
        """
           (SLO Breached)
        ---------------------- High water mark


        ---------------------- Low water mark
           (SLO Unbreached)
        """
        p = self.opc.get_pth_percentile()

        # If we breached the high watermark
        # Let Controller know => MPS to MIG might take place
        if p >= self.slo_high_wm:
            if not self.ctrl_slo_comm_running.get():
                self.ctrl_slo_comm_running.flip()
                slo_thread = threading.Thread(
                    target=self._convey_slo_breach, args=(WatermarkType.HIGH,)
                )
                slo_thread.daemon = True
                slo_thread.start()

        # If we go below the low watermark
        # Let Controller know => MIG to MPS might take place
        elif p <= self.slo_low_wm:
            if not self.ctrl_slo_comm_running.get():
                self.ctrl_slo_comm_running.flip()
                slo_thread = threading.Thread(
                    target=self._convey_slo_breach, args=(WatermarkType.LOW,)
                )
                slo_thread.daemon = True
                slo_thread.start()

    def dequeue_deferred_thread(self, queue):
        sysvq = SysVQueue(self.qid, create_new_queue=False)
        rcvd_items = sysvq.read_queue(queue)
        print(f"Received {rcvd_items} from queue-{self.qid} (tid={self.tid})",
              file=sys.stderr)

    def run_infer_executor(self, num_reqs, i=0, is_warmup=False):
        # Create a queue to enqueue requests (for non closed-loop experiment)
        enqueue_thread = None
        if self.distribution_type != DistributionType.CLOSED:
            queue = Queue()
            enqueue_thread = threading.Thread(
                target=self.enqueue_requests,
                args=(queue, num_reqs, is_warmup)
            )
            enqueue_thread.start()

        # Enqueue any deferred request to the process
        if (not is_warmup and self.qid != -1 and
            self.distribution_type != DistributionType.CLOSED
        ):
            deferred_dequeue_thread = threading.Thread(
                target=self.dequeue_deferred_thread,
                args=(queue,)
            )
            deferred_dequeue_thread.daemon = True
            deferred_dequeue_thread.start()

        completed = 0
        total_time_arr = []
        queued_time_arr = []

        is_percentile_calc = (not is_warmup) and (self.ctrl_grpc_channel)
        process_start_time = time.time()
        while i < num_reqs:
            if self.job_completed:
                break
            try:
                if self.distribution_type != DistributionType.CLOSED:
                    queued_time = queue.get(block=True, timeout=1)
                else:
                    queued_time = time.time()
            except QueueEmptyException:
                continue
            start_time = time.time()
            completed += self.model_obj.infer()
            end_time = time.time()

            total_time_arr.append(end_time - queued_time)
            queueing_delay = start_time - queued_time
            queued_time_arr.append(queueing_delay)

            if is_percentile_calc:
                self.opc.add_element(queueing_delay)

            if self._orion_lib:
                orion_block(self._orion_lib, i)
            i += 1
        process_end_time = time.time()

        if enqueue_thread:
            enqueue_thread.join()
        return [
            completed / (process_end_time - process_start_time),
            total_time_arr,
            queued_time_arr
        ]

    def install_signal_handler(self):
        signal.signal(signal.SIGUSR1, self._catch_to_start)
        signal.signal(signal.SIGUSR2, self._catch_to_end)

    def _catch_to_end(self, signum, frame):
        self.finish = True
        if self.distribution_type == DistributionType.CLOSED:
            self.job_completed = True

    def _catch_to_start(self, signum, frame):
        self.start = True

    def enqueue_requests(self, queue, num_reqs, is_warmup):
        sleep_array_len = len(self._sleep_time)
        is_percentile_calc = (not is_warmup) and (self.ctrl_grpc_channel)
        slo_check_every = SLO_HISTORY_BUFFER_SIZE / 4
        for i in range(num_reqs):
            queued_time = time.time()
            queue.put(queued_time)
            if self.finish:
                break

            curr = timeit.default_timer()
            if is_percentile_calc and i % slo_check_every == 0:
                self._slo_check()
            time.sleep(
                max(0,
                    self._sleep_time[i % sleep_array_len] -
                        (timeit.default_timer() - curr)
                )
            )

        # Before exiting, dump all items of the queue
        if not is_warmup:
            if not self.ctrl_grpc_channel:
                while not queue.empty():
                    queue.get()
            else:
                pid = os.getpid()
                sysvq = SysVQueue(pid, create_new_queue=True)
                successful_sends = sysvq.dump_queue(queue)
                print(
                    f"Sent {successful_sends} items on queue-{pid} (tid={self.tid})",
                    file=sys.stderr
                )
            self.job_completed = True

    def _retire_experiment(self):
        time.sleep(self.duration)
        self.finish = True

    def _indicate_ready(self):
        # Wait till user instructs to start
        # This can be either:
        # 1) Via barrier sync: For multi threads
        # 2) Via signal: For multi process
        if self._thread_block():
            self.start = True
        else:
            # Install Signal Handler
            self.install_signal_handler()

            # Write to indicate readiness
            # there by the user can signal when all procs are ready
            log(f"/tmp/{os.getpid()}", "")
            while not self.start:
                pass

        # As we get ready to run experiment, setup retire time if needed
        if self.duration > 0:
            retire_exp_thread = threading.Thread(
                target=self._retire_experiment,
            )
            retire_exp_thread.daemon = True
            retire_exp_thread.start()

    def _return_infer_stats(self, infer_stats):
        infer_stats.insert(0, f"{self.model_obj.get_id()}")
        result = (self.tid, infer_stats)
        if self.return_queue:
            self.return_queue.put(result)
        else:
            with open(f"/tmp/{os.getpid()}.pkl", "wb") as h:
                pickle.dump(result, h)

    def _thread_block(self):
        if self.thread_barrier:
            self.thread_barrier.wait()
            return True
        return False

    def run(self):
        # Ready to load
        self._thread_block()

        # Load Model
        if self.thread_barrier:
            time.sleep(random.randint(1, 5))
        self.model_obj.load_model()

        # Read input to infer
        self.model_obj.load_data()

        # Warm up the model
        self.run_infer_executor(WARMUP_REQS, is_warmup=True)
        reqs_completed = WARMUP_REQS

        # Ready for experiment
        self._indicate_ready()

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("start")
        infer_stats = self.run_infer_executor(
            self.num_infer + reqs_completed,
            i=reqs_completed,
            is_warmup=False
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

        # Give stats back to user
        self._return_infer_stats(infer_stats)


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--model-type", type=str, default="vision")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-infer", type=int, default=sys.maxsize)
    parser.add_argument("--distribution-type", type=str, default="closed")
    parser.add_argument("--rps", type=float, default=30)
    parser.add_argument("--tid", type=int, default=0)
    parser.add_argument("--qid", type=int, default=-1)
    parser.add_argument("--slo-percentile", type=float, default=90)
    parser.add_argument("--slo-lw", type=float, default=0)
    parser.add_argument("--slo-hw", type=float, default=float("inf"))
    parser.add_argument("--ctrl-grpc", type=str, default=None)
    opt, unused_args = parser.parse_known_args()

    # Create batched inference object
    model_obj = get_batched_inference_object(
        opt.model_type,
        opt.device_id,
        opt.model,
        opt.batch_size
    )

    # Create executor object
    if opt.ctrl_grpc == "null" or not opt.ctrl_grpc:
        opt.ctrl_grpc = None
    executor_obj = BatchedInferenceExecutor(
        model_obj,
        opt.distribution_type,
        opt.rps,
        opt.tid,
        qid=opt.qid,
        num_infer=opt.num_infer,
        slo_percentile=opt.slo_percentile,
        slo_lw=opt.slo_lw,
        slo_hw=opt.slo_hw,
        ctrl_grpc_channel=opt.ctrl_grpc,
    )

    # Run experiment
    executor_obj.run()
