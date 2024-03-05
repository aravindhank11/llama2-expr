import os
import threading
import pickle
import argparse
import signal
import sys
import time
import random
import torch
from ctypes import cdll
from enum import Enum
from queue import Queue
from queue import Empty as QueueEmptyException
import numpy as np
from batched_inference import get_batched_inference_object


WARMUP_REQS = 10
LARGE_NUM_REQS = 100000
MAX_QUEUE_SIZE = 1000
ORION_LIB = "/root/orion/src/cuda_capture/libinttemp.so"


def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)


class DistributionType(Enum):
    CLOSED = (1, "CLOSED")
    POINT = (2, "POINT")
    POISSON = (3, "POISSON")


def block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)


class BatchedInferenceExecutor:
    def __init__(
        self, model_obj, distribution_type, rps, tid,
        num_infer=LARGE_NUM_REQS,
        thread_barrier=None,
        return_queue=None,
        duration=-1
    ):
        self.model_obj = model_obj
        self.num_infer = num_infer
        self.rps = rps
        self.duration = duration

        # Thread synchronization mechanism
        self.tid = tid
        self.thread_barrier = thread_barrier
        self.return_queue = return_queue

        # Process synchronization mechanism
        self.start = False
        self.finish = False

        self._orion_lib = None
        self._reqs_completed = 0
        if os.path.exists(ORION_LIB):
            self._orion_lib = cdll.LoadLibrary(ORION_LIB)

        if distribution_type == "closed":
            self.distribution_type = DistributionType.CLOSED
        elif distribution_type == "point":
            self.distribution_type = DistributionType.POINT
            self._sleep_time = [1/rps]
        elif distribution_type == "poisson":
            self.distribution_type = DistributionType.POISSON
            self._sleep_time = np.random.exponential(
                scale=1/rps,
                size=10000
            )
        else:
            print(f"Unknown distribution type: {distribution_type}")
            print("Allowed values: closed, point, poisson")
            sys.exit(1)

    def run_infer_executor(self, num_reqs):
        # Create a queue to enqueue requests (for non closed-loop experiment)
        enqueue_thread = None
        if self.distribution_type != DistributionType.CLOSED:
            queue = Queue()
            enqueue_thread = threading.Thread(
                target=self.enqueue_requests,
                args=(queue, num_reqs)
            )
            enqueue_thread.start()

        completed = 0
        total_time_arr = []
        queued_time_arr = []
        i = 0

        process_start_time = time.time()
        while i < num_reqs:
            if self.finish:
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
            queued_time_arr.append(start_time - queued_time)
            if self._orion_lib:
                block(self._orion_lib, i + self._reqs_completed)
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

    def _catch_to_start(self, signum, frame):
        self.start = True

    def enqueue_requests(self, queue, num_reqs):
        sleep_array_len = len(self._sleep_time)
        for i in range(num_reqs):
            queued_time = time.time()
            if queue.qsize() < MAX_QUEUE_SIZE:
                queue.put(queued_time)
            else:
                print("ALERT! Queue overflow!!!")

            if self.finish:
                break
            time.sleep(self._sleep_time[i % sleep_array_len])

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
        self.run_infer_executor(WARMUP_REQS)
        self._reqs_completed += WARMUP_REQS

        # Ready for experiment
        self._indicate_ready()

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("start")
        infer_stats = self.run_infer_executor(
            max(1, self.num_infer - self._reqs_completed)
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

        # Give stats back to user
        self._return_infer_stats(infer_stats)


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-name", type=str, default="cuda:0")
    parser.add_argument("--model-type", type=str, default="vision")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-infer", type=int, default=LARGE_NUM_REQS)
    parser.add_argument("--distribution_type", type=str, default="closed")
    parser.add_argument("--rps", type=float, default=30)
    parser.add_argument("--tid", type=int, default=0)
    opt = parser.parse_args()

    # Create batched inference object
    model_obj = get_batched_inference_object(
        opt.model_type,
        opt.device_name,
        opt.model,
        opt.batch_size
    )

    # Create executor object
    executor_obj = BatchedInferenceExecutor(
        model_obj,
        opt.distribution_type,
        opt.rps,
        opt.tid,
        opt.num_infer
    )

    # Run experiment
    executor_obj.run()
