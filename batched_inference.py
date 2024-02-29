# Ref: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
import os
import threading
import pickle
import pathlib
import argparse
from queue import Queue
from queue import Empty as QueueEmptyException
import signal
import time
import sys
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
from transformers import LlamaTokenizer, LlamaForCausalLM
from ctypes import cdll

WARMUP_REQS = 10
LARGE_NUM_REQS = 100000
MAX_QUEUE_SIZE = 1000
ORION_LIB = "/root/orion/src/cuda_capture/libinttemp.so"


def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)


def block(backend_lib, it):
    # block client until request served
    backend_lib.block(it)


class DistributionType(Enum):
    CLOSED = (1, "CLOSED")
    POINT = (2, "POINT")
    POISSON = (3, "POISSON")


class BatchedInference(ABC):
    def __init__(self, device_name, model_name, batch_size):
        self._device = torch.device(device_name)
        self._model_name = model_name
        self._model = None
        self._batch_size = batch_size

    @abstractmethod
    def get_id(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def infer(self):
        pass

class VisionBatchedInference(BatchedInference):
    def __init__(self, device_name, model_name, batch_size):
        super().__init__(device_name, model_name, batch_size)
        self._imgs = None

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        try:
            self._model = torch.hub.load(
                "pytorch/vision:v0.14.1",
                self._model_name,
                verbose=False,
                pretrained=True
            )
        except ImportError:
            # Orion profiler hack
            self._model = models.__dict__[self._model_name](num_classes=1000)

        self._model.eval()
        self._model.to(self._device)

    def __load_numpy_data(self, image):
        preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])

        input_tensor = preprocess(image)
        imgs = input_tensor.unsqueeze(0)
        self._imgs = imgs.repeat(self._batch_size, 1, 1, 1).pin_memory()

    def __load_non_numpy_data(self, image):
        # Orion profiler hack
        image = image.resize((256, 256))
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        right = (256 + 224) // 2
        bottom = (256 + 224) // 2
        image = image.crop((left, top, right, bottom))

        # Convert PIL image to tensor manually
        tensor_image = torch.tensor(
            [list(image.getdata())], dtype=torch.float32
        )
        # Reshape to CHW format
        tensor_image = tensor_image.view(1, 3, 224, 224)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor_image = (
            tensor_image - mean.view(1, 3, 1, 1)
        ) / std.view(1, 3, 1, 1)

        # Repeat the image batch_size times
        self._imgs = tensor_image.repeat(self._batch_size, 1, 1, 1).pin_memory()

    def load_data(self):
        curr_path = pathlib.Path(__file__).parent.resolve()
        image_path = os.path.join(curr_path, "data/images/dog.jpg")
        image = Image.open(image_path)

        try:
            self.__load_numpy_data(image)
        except RuntimeError:
            # Orion docker's torch is compiled without numpy support :(
            self.__load_non_numpy_data(image)

    def infer(self):
        to_infer = self._imgs.to(self._device, non_blocking=True)
        with torch.no_grad():
            output = self._model(to_infer)
        #torch.cuda.synchronize()
        return self._batch_size


class LlamaBatchedInference(BatchedInference):
    def __init__(self, device_name, model_name, batch_size):
        super().__init__(device_name, model_name, batch_size)
        self.prompts = [
            "A short summary on the hottest researched topic",
            "What are the implications of AI on society",
            "Imagine a future with no human beings",
            "Discuss the current state of renewable energy"
        ]
        self.model_path = "openlm-research/open_llama_7b_v2"
        self._llama_prompt_ctr = -1
        self._input_tokens = []
        self._num_input_tokens = 0

    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        ).to(self._device)

    def load_data(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        for prompt in self.prompts:
            self._input_tokens.append(
                tokenizer(prompt, return_tensors="pt").input_ids
            )
        self._num_input_tokens = len(self._input_tokens)

    def infer(self):
        self._llama_prompt_ctr += 1
        input_tokens = (self._input_tokens[self._llama_prompt_ctr % self._num_input_tokens]
                .to(self._device, non_blocking=True)
        )
        generation_output = self._model.generate(
            input_ids=input_tokens,
            max_new_tokens=self._batch_size
        )
        return generation_output.size()[1]


class BatchedInferenceExecutor:
    def __init__(
        self, model_obj, distribution_type, rps,
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
        total_time = 0
        total_time_arr = []
        queued_time_arr = []
        for i in range(num_reqs):
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

            total_time += end_time - queued_time
            total_time_arr.append(end_time - queued_time)
            queued_time_arr.append(start_time - queued_time)
            if self._orion_lib:
                block(self._orion_lib, i + self._reqs_completed)

        if enqueue_thread:
            enqueue_thread.join()
        return [completed / total_time, total_time_arr, queued_time_arr]


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
        if self.return_queue:
            self.return_queue.put(infer_stats)
        else:
            with open(f"/tmp/{os.getpid()}.pkl", "wb") as h:
                pickle.dump(infer_stats, h)

    def _thread_block(self):
        if self.thread_barrier:
            self.thread_barrier.wait()
            return True
        return False

    def run(self):
        # Ready to load
        self._thread_block()

        # Load Model
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


def get_batched_inference_object(model_type, device_name, model, batch_size):
    if model_type == "vision":
        return VisionBatchedInference(device_name, model, batch_size)
    elif model_type == "llama":
        return LlamaBatchedInference(device_name, model, batch_size)
    else:
        print("Unknown model-type. Must be one of 'vision', 'llama'")
        sys.exit(1)

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
        opt.num_infer
    )

    # Run experiment
    executor_obj.run()
