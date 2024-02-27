# Ref: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

import torch
from PIL import Image
import numpy as np

# Import to profile
from torchvision import transforms, models
from transformers import LlamaTokenizer, LlamaForCausalLM

import os
import threading
import pickle
import pathlib
import argparse
import traceback
from queue import Queue
import signal
import time
import sys

WARMUP_REQS = 5
MAX_QUEUE_SIZE = 1000

PROMPTS = [
    "A short summary on the hottest researched topic",
    "What are the implications of AI on society",
    "Imagine a future with no human beings",
    "Discuss the current state of renewable energy"
]
LLAMA_MODEL_PATH = "openlm-research/open_llama_7b_v2"


def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)


class BatchedInference:
    def __init__(self, device_name, model_name, num_infer, distribution_type, rps):
        self._device = torch.device(device_name)
        self._model_name = model_name
        self.start = False
        self.finish = False
        self.num_infer = num_infer
        self._llama_prompt_ctr = -1
        self.distribution_type=distribution_type
        self.rps = rps
        if distribution_type == "closed":
            self._sleep_time = [0]
        elif distribution_type == "point":
            self._sleep_time = [1/rps]
        elif distribution_type == "poisson":
            self._sleep_time = np.random.exponential(
                scale=1/rps,
                size=10000
            )
        else:
            print(f"Unknown distribution type: {distribution_type}")
            print("Allowed values: closed, point, poisson")
            sys.exit(1)


    ################## LOAD MODELS ###################
    def __load_llama_model(self):
        self._model = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_PATH, torch_dtype=torch.float16 #, device_map="auto",
        ).to(self._device)
        return True

    def __load_img_model(self):
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
        return True

    def load_model(self):
        try:
            if self._model_name == "llama":
                return self.__load_llama_model()
            return self.__load_img_model()
        except:
            print(traceback.format_exc())
            return False

    ################## LOAD INPUTS ###################
    def __load_llama_input(self):
        self._input_tokens = []
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL_PATH)
        for prompt in PROMPTS:
            self._input_tokens.append(
                tokenizer(prompt, return_tensors="pt").input_ids
            )
        self._num_input_tokens = len(self._input_tokens)
        return True

    def __load_img_input(self):
        curr_path = pathlib.Path(__file__).parent.resolve()

        # Images for standard torch models
        image_path = os.path.join(curr_path, "data/images/dog.jpg")
        image = Image.open(image_path)

        try:
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
            self._imgs = imgs.repeat(self.batch_size, 1, 1, 1).pin_memory()

        except RuntimeError:
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
            self._imgs = tensor_image.repeat(self.batch_size, 1, 1, 1).pin_memory()

        return True

    def load_input(self, batch_size):
        try:
            self.batch_size = batch_size
            if self._model_name == "llama":
                return self.__load_llama_input()
            return self.__load_img_input()
        except:
            print(traceback.format_exc())
            return False

    ################## LOAD INPUTS ###################
    def __infer_llama(self):
        self._llama_prompt_ctr += 1
        input_tokens = (self._input_tokens[self._llama_prompt_ctr % self._num_input_tokens]
                .to(self._device, non_blocking=True)
        )
        generation_output = self._model.generate(
            input_ids=input_tokens,
            max_new_tokens=self.batch_size
        )
        return generation_output.size()[1]

    def __infer_img(self):
        to_infer = self._imgs.to(self._device, non_blocking=True)
        with torch.no_grad():
            output = self._model(to_infer)
        torch.cuda.synchronize()
        return self.batch_size

    def infer(self, num_reqs):
        try:
            queue = Queue()
            enqueue_thread = threading.Thread(
                target=self.enqueue_requests,
                args=(queue, num_reqs)
            )
            enqueue_thread.start()

            if self._model_name == "llama":
                infer_fn = self.__infer_llama
            else:
                infer_fn = self.__infer_img

            completed = 0
            total_time = 0
            total_time_arr = []
            queued_time_arr = []
            for _ in range(num_reqs):
                if self.finish:
                    break

                try:
                    queued_time = queue.get(block=True, timeout=1)
                except queue.Empty:
                    continue
                start_time = time.time()
                completed += infer_fn()
                end_time = time.time()

                total_time += end_time - queued_time
                total_time_arr.append(end_time - queued_time)
                queued_time_arr.append(start_time - queued_time)

            enqueue_thread.join()
            return [completed / total_time, total_time_arr, queued_time_arr]
        except:
            print(traceback.format_exc())
            return 0

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
            if (self.finish):
                break
            time.sleep(self._sleep_time[i % sleep_array_len])

    def run(self):
        while not self.start:
            pass

        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("start")
        infer_stats = self.infer(self.num_infer)
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

        infer_stats.insert(0, f"{self._model_name}-{self.batch_size}-{self.distribution_type}-{self.rps}")
        with open(f"/tmp/{os.getpid()}.pkl", "wb") as h:
            pickle.dump(infer_stats, h)
        sys.exit(0)


if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-name", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-infer", type=int, default=100000)
    parser.add_argument("--distribution_type", type=str, default="closed")
    parser.add_argument("--rps", type=int, default=30)
    opt = parser.parse_args()

    obj = BatchedInference(
        opt.device_name, opt.model, opt.num_infer,
        opt.distribution_type, opt.rps
    )

    # Try loading model, if model load fails -> exit
    success = obj.load_model()
    if not success:
        log("/tmp/{}_oom".format(os.getpid()), "")
        sys.exit(1)

    # Read images to infer
    obj.load_input(opt.batch_size)

    # Warm up the model
    # If out of memory while infering -> exit
    obj.infer(WARMUP_REQS)

    # Install Signal Handler
    obj.install_signal_handler()

    # Write to indicate load is success
    log("/tmp/{}".format(os.getpid()), "")

    # Wait for signal to start inference
    obj.run()
