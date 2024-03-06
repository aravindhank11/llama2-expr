from ctypes import (
    cdll,
    c_int,
    c_char_p,
    c_bool,
    c_void_p,
    POINTER
)
import argparse
import os
import pickle
from queue import Queue
from threading import Barrier, Thread

from batched_inference_executor import (
    BatchedInferenceExecutor,
    LARGE_NUM_REQS,
    WARMUP_REQS
)
from batched_inference import (
    get_batched_inference_object,
)


ORION_SCHEDULER_LIB_PATH = "/root/orion/src/scheduler/scheduler_eval.so"


class ModelDetails:
    def __init__(self, device_type, model_description):
        md_list = model_description.split("-")
        self.device_type = device_type
        self.model_type = self._parse(md_list, "model_type", 0, str)
        self.model_name = self._parse(md_list, "model_name", 1, str)
        self.batch_size = self._parse(md_list, "batch_size", 2, int)
        self.distribution_type = self._parse(
            md_list, "distribution_type", 3, str
        )
        self.rps = self._parse(md_list, "rps", 4, float)
        self._get_kernel_details()

    def _parse(self, md_list, id_, pos, type_):
        if len(md_list) < pos + 1:
            raise ValueError(f"Unable to parse '{id_}'")
        val = md_list[pos]

        try:
            return type_(val)
        except:
            raise ValueError(f"Unable to convert {id_} to '{type_}'")

    def _get_kernel_details(self):
        self.kernel_file = (
            f"orion-fork/results/{self.device_type}/" +
            f"{self.model_name}/batchsize-{self.batch_size}/" +
            "orion_input.csv"
        )
        if not os.path.exists(self.kernel_file):
            raise ValueError(f"{self.kernel_file} does not exist")

        with open(self.kernel_file, "r") as file:
            self.num_kernels = sum(1 for _ in enumerate(file, start=1))
            self.num_kernels -= 1


def run_object(obj):
    obj.run()


def run_scheduler(barrier, threads, model_names, kernel_files, num_kernels):
    # Get thread ID from thread
    tids = [t.native_id for t in threads]
    num_clients = len(tids)

    # Define C Types
    int_arr_t = c_int * num_clients
    char_arr_t = c_char_p * num_clients
    bool_arr_t = c_bool * num_clients

    # convert to c_types
    c_tids = int_arr_t(*tids)
    c_num_kernels = int_arr_t(*num_kernels)
    c_num_iters = int_arr_t(*([LARGE_NUM_REQS] * num_clients))
    c_model_names = char_arr_t(*model_names)
    c_kernel_files = char_arr_t(*kernel_files)
    train_ar = bool_arr_t(*([False] * num_clients))

    # Setup library
    sched_lib = cdll.LoadLibrary(ORION_SCHEDULER_LIB_PATH)
    scheduler = sched_lib.sched_init()
    sched_lib.argtypes = [
        c_void_p, c_int, POINTER(c_int), POINTER(c_char_p),
        POINTER(c_char_p), POINTER(c_int), POINTER(c_bool)
    ]
    sched_lib.setup(
        scheduler, num_clients, c_tids, c_model_names, c_kernel_files,
        c_num_kernels, c_num_iters, train_ar, False
    )
    barrier.wait()

    # Warmup
    sched_lib.schedule(
        scheduler,       # Scheduler
        num_clients,     # Number of parallel jobs to run
        True,            # profile_mode
        0,               # iter
        True,            # warmup
        WARMUP_REQS,     # warmup_iters
        False,           # reef
        False,           # sequential
        1,               # reef_depth
        1,               # hp_limit
        1                # update_start
    )
    barrier.wait()

    # Actual
    sched_lib.schedule(
        scheduler,      # Scheduler
        num_clients,    # Number of parallel jobs to run
        True,           # profile_mode
        0,              # iter
        False,          # warmup
        0,              # warmup_iters
        False,          # reef
        False,          # sequential
        1,              # reef_depth
        1,              # hp_limit
        1               # update_start
    )
    barrier.wait()


def spin_orion_scheduler(device_id, duration, md_list):
    num_clients = len(md_list)
    barrier = Barrier(num_clients + 1)
    threads = []
    model_names = []
    kernel_files = []
    num_kernels = []
    result_queue = Queue()

    for tid, md in enumerate(md_list):
        model_obj = get_batched_inference_object(
            md.model_type,
            device_id,
            md.model_name,
            md.batch_size
        )

        executor_obj = BatchedInferenceExecutor(
            model_obj, md.distribution_type, md.rps, tid,
            thread_barrier=barrier, return_queue=result_queue,
            duration=duration
        )

        t = Thread(target=run_object, args=(executor_obj,))
        t.start()
        threads.append(t)
        model_names.append(md.model_name.encode("utf-8"))
        kernel_files.append(md.kernel_file.encode("utf-8"))
        num_kernels.append(md.num_kernels)

    schedule_thread = Thread(
        target=run_scheduler,
        args=(
            barrier, threads, model_names, kernel_files, num_kernels
        )
    )
    schedule_thread.daemon = True
    schedule_thread.start()

    for i, t in enumerate(threads):
        t.join()
        result = result_queue.get()
        with open(f"/tmp/{os.getpid()}-{i}.pkl", "wb") as h:
            pickle.dump(result, h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-type", type=str, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument(
        "model_details",
        metavar="<model_type>-<model_name>-<batch_size>-<distribution_type>-<rps>",
        nargs="+",
        help="List of model and details"
    )
    opt = parser.parse_args()

    model_details = []
    for model_detail in opt.model_details:
        model_details.append(ModelDetails(opt.device_type, model_detail))

    spin_orion_scheduler(opt.device_id, opt.duration, model_details)
