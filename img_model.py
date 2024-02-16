#Ref: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

import torch
from PIL import Image

# Import to profile
from torchvision import transforms

import os
import pathlib
import argparse
import traceback
from collections import defaultdict

import signal
import time
import sys

WARMUP_REQS = 5


def log(filename, string):
    with open(filename, "w") as h:
        h.write(string)

class BatchedInference:
    def __init__(self, device_name, model_name):
        self._device = torch.device(device_name)
        self._model_name = model_name
        self.finish = False

    def load_model(self):
        try:
            self._model = torch.hub.load("pytorch/vision:v0.14.1",
                                          self._model_name,
                                          verbose=False,
                                          pretrained=True)
            self._model.eval()
            self._model.to(self._device)
            return True

        except:
            print(traceback.format_exc())
            return False

    def read_images(self, batch_size):
        self.batch_size = batch_size

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

    def infer(self, num_reqs):

        total_time = 0
        reqs = 0

        for _ in range(num_reqs):
            to_infer = self._imgs.to(self._device, non_blocking=True)

            # Actual Inference
            try:
                start_time = time.time()
                with torch.no_grad():
                    output = self._model(to_infer)
                torch.cuda.synchronize()

                total_time += time.time() - start_time
                reqs += 1
                if (self.finish):
                    break
            except:
                print(traceback.format_exc())
                return 0

        return (reqs * self.batch_size) / total_time

    def install_signal_handler(self):
        signal.signal(signal.SIGUSR1, self._catch)
        signal.signal(signal.SIGUSR2, self._catch_to_end)

    def _catch_to_end(self, signum, frame):
        self.finish = True

    def _catch(self, signum, frame):
        torch.cuda.cudart().cudaProfilerStart()
        rps = self.infer(10000000)
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
    obj.read_images(opt.batch_size)

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
