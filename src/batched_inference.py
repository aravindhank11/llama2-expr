# Ref: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
import sys
import os
import pathlib
from abc import ABC, abstractmethod
from PIL import Image
import torch
from torchvision import transforms, models
from transformers import LlamaTokenizer, LlamaForCausalLM


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
            # Orion profiler hack
            self._model = models.__dict__[self._model_name](num_classes=1000)
        except:
            self._model = torch.hub.load(
                "pytorch/vision:v0.14.1",
                self._model_name,
                verbose=False,
                pretrained=True
            )

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
        self._imgs = tensor_image.repeat(
            self._batch_size, 1, 1, 1
        ).pin_memory()

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
        torch.cuda.synchronize()
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
        input_tokens = (
            self._input_tokens[self._llama_prompt_ctr % self._num_input_tokens]
            .to(self._device, non_blocking=True)
        )
        generation_output = self._model.generate(
            input_ids=input_tokens,
            max_new_tokens=self._batch_size
        )
        return generation_output.size()[1]


def get_batched_inference_object(model_type, device_name, model, batch_size):
    if model_type == "vision":
        return VisionBatchedInference(device_name, model, batch_size)
    elif model_type == "llama":
        return LlamaBatchedInference(device_name, model, batch_size)
    else:
        print("Unknown model-type. Must be one of 'vision', 'llama'")
        sys.exit(1)
