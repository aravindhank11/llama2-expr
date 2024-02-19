#Ref: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

import torch
from PIL import Image

# Import to profile
from torchvision import transforms
from transformers import LlamaTokenizer, LlamaForCausalLM

import os
import pickle
import pathlib
import argparse
import traceback
from collections import defaultdict

import signal
import time
import sys

WARMUP_REQS = 5

PROMPTS = [
    "A short summary on the hottest researched topic",
    "What are the implications of AI on society",
    "Imagine a future with no human beings",
    "Discuss the current state of renewable energy"
]
LLAMA_MODEL_PATH = 'openlm-research/open_llama_7b_v2'


def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)

class BatchedInference:
    def __init__(self, device_name, model_name):
        self._device = torch.device(device_name)
        self._model_name = model_name
        self.finish = False

    ################## LOAD MODELS ###################
    def __load_llama_model(self):
        self._model = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_PATH, torch_dtype=torch.float16 #, device_map='auto',
        ).to(self._device)
        return True

    def __load_img_model(self):
        self._model = torch.hub.load("pytorch/vision:v0.14.1",
                                      self._model_name,
                                      verbose=False,
                                      pretrained=True)
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
            self._input_tokens.append(tokenizer(prompt, return_tensors="pt").input_ids)
        return True

    def __load_img_input(self):
        curr_path = pathlib.Path(__file__).parent.resolve()

        # Images for standard torch models
        image_path = os.path.join(curr_path, "data/images/dog.jpg")
        img = Image.open(image_path)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(img)
        imgs = input_tensor.unsqueeze(0)
        self._imgs = imgs.repeat(self.batch_size, 1, 1, 1).pin_memory()

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
    def __infer_llama(self, num_reqs):
        total_time = 0
        tokens_generated = 0
        num_input_tokens = len(self._input_tokens)
        time_taken_arr = []

        for i in range(num_reqs):
            start_time = time.time()

            input_tokens = (self._input_tokens[i % num_input_tokens]
                    .to(self._device, non_blocking=True)
            )
            generation_output = self._model.generate(
                input_ids=input_tokens,
                max_new_tokens=self.batch_size
            )

            time_taken = time.time() - start_time
            time_taken_arr.append(time_taken)
            total_time += time_taken
            tokens_generated += generation_output.size()[1]

            if (self.finish):
                break

        return tokens_generated / total_time, time_taken_arr

    def __infer_img(self, num_reqs):
        total_time = 0
        reqs = 0
        time_taken_arr = []

        for _ in range(num_reqs):
            start_time = time.time()

            to_infer = self._imgs.to(self._device, non_blocking=True)
            with torch.no_grad():
                output = self._model(to_infer)
            torch.cuda.synchronize()

            time_taken = time.time() - start_time
            time_taken_arr.append(time_taken)
            total_time += time_taken
            reqs += 1

            if (self.finish):
                break

        return (reqs * self.batch_size) / total_time, time_taken_arr

    def infer(self, num_reqs):
        try:
            if self._model_name == "llama":
                return self.__infer_llama(num_reqs)
            return self.__infer_img(num_reqs)
        except:
            print(traceback.format_exc())
            return 0

    def install_signal_handler(self):
        signal.signal(signal.SIGUSR1, self._catch)
        signal.signal(signal.SIGUSR2, self._catch_to_end)

    def _catch_to_end(self, signum, frame):
        self.finish = True

    def _catch(self, signum, frame):
        torch.cuda.cudart().cudaProfilerStart()
        rps, time_taken = self.infer(10000000)
        with open(f"/tmp/{os.getpid()}.pkl", "wb") as h:
            pickle.dump(time_taken, h)
        log("/tmp/{}".format(os.getpid()), "{:.2f}".format(rps))
        torch.cuda.cudart().cudaProfilerStop()
        sys.exit(0)

if __name__ == "__main__":

    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-name', type=str, default="cuda:0")
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--batch-size', type=int, required=True)
    opt = parser.parse_args()

    obj = BatchedInference(opt.device_name, opt.model)

    # Try loading model, if model load fails -> exit
    success = obj.load_model()
    if not(success):
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

    # Sleep for ever
    while(True):
        time.sleep(1000000)
