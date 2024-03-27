# Ref: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
import random
import sys
import os
import pathlib
from abc import ABC, abstractmethod
from PIL import Image
import torch
from torchvision import transforms, models
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, BertForQuestionAnswering


class BatchedInference(ABC):
    def __init__(self, device_id, model_name, batch_size):
        self._device = torch.device(f"cuda:{device_id}")
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
    def __init__(self, device_id, model_name, batch_size):
        super().__init__(device_id, model_name, batch_size)
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
    def __init__(self, device_id, model_name, batch_size):
        super().__init__(device_id, model_name, batch_size)
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
            max_new_tokens=self._batch_size,
            do_sample=True
        )
        return generation_output.size()[1]

class BertBatchedInference(BatchedInference):
    def __init__(self, device_id, model_name, batch_size):
        super().__init__(device_id, model_name, batch_size)
        self.prompts = [
            "What is the capital of France?",
            "Who wrote 'Romeo and Juliet'?",
            "What is the tallest mountain in the world?",
            "Who discovered electricity?",
            "What is the chemical formula for water?",
            "Who painted the Mona Lisa?",
            "What is the currency of Japan?",
            "Who is the current president of the United States?",
            "What is the largest planet in our solar system?",
            "Who invented the telephone?",
            "What is the currency of Australia?",
            "Who was the first man to walk on the moon?",
            "What is the speed of light?",
            "Who is the author of 'To Kill a Mockingbird'?",
            "What is the main ingredient in sushi?",
            "Who composed the 'Moonlight Sonata'?",
            "What is the population of China?",
            "Who was the first woman to win a Nobel Prize?",
            "What is the atomic number of carbon?",
            "Who is the CEO of Tesla?",
            "What is the national animal of Australia?",
            "Who was the first President of the United States?",
            "What is the largest ocean on Earth?",
            "Who discovered penicillin?",
            "What is the boiling point of water?",
            "Who painted 'Starry Night'?",
            "What is the capital of Russia?",
            "Who wrote '1984'?",
            "What is the longest river in the world?",
            "Who developed the theory of relativity?",
            "What is the tallest building in the world?",
            "Who is the current Prime Minister of the United Kingdom?",
            "What is the chemical symbol for gold?",
            "Who founded Microsoft?",
            "What is the temperature at which Fahrenheit and Celsius scales are equal?",
            "Who is the Greek god of the sea?",
            "What is the capital of Brazil?",
            "Who directed the movie 'The Shawshank Redemption'?",
            "What is the largest desert in the world?",
            "Who was the first female Prime Minister of the United Kingdom?"
        ]

        self.contexts = [
            "The capital city of France is Paris.",
            "'Romeo and Juliet' was written by William Shakespeare.",
            "Mount Everest is the tallest mountain in the world.",
            "Electricity was discovered by Benjamin Franklin.",
            "The chemical formula for water is H2O.",
            "The Mona Lisa was painted by Leonardo da Vinci.",
            "The currency of Japan is the Japanese yen.",
            "The current president of the United States is Joe Biden.",
            "Jupiter is the largest planet in our solar system.",
            "The telephone was invented by Alexander Graham Bell.",
            "The currency of Australia is the Australian dollar.",
            "The first man to walk on the moon was Neil Armstrong.",
            "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
            "Harper Lee is the author of 'To Kill a Mockingbird'.",
            "The main ingredient in sushi is rice.",
            "Ludwig van Beethoven composed the 'Moonlight Sonata'.",
            "The population of China is over 1.4 billion people.",
            "Marie Curie was the first woman to win a Nobel Prize.",
            "The atomic number of carbon is 6.",
            "Elon Musk is the CEO of Tesla.",
            "The national animal of Australia is the kangaroo.",
            "George Washington was the first President of the United States.",
            "The largest ocean on Earth is the Pacific Ocean.",
            "Alexander Fleming discovered penicillin.",
            "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit.",
            "Vincent van Gogh painted 'Starry Night'.",
            "The capital of Russia is Moscow.",
            "George Orwell wrote '1984'.",
            "The longest river in the world is the Nile River.",
            "Albert Einstein developed the theory of relativity.",
            "The tallest building in the world is the Burj Khalifa in Dubai, United Arab Emirates.",
            "The current Prime Minister of the United Kingdom is Boris Johnson.",
            "The chemical symbol for gold is Au.",
            "Microsoft was founded by Bill Gates and Paul Allen.",
            "The temperature at which Fahrenheit and Celsius scales are equal is -40 degrees.",
            "Poseidon is the Greek god of the sea.",
            "The capital of Brazil is Brasília.",
            "Frank Darabont directed the movie 'The Shawshank Redemption'.",
            "The largest desert in the world is the Sahara Desert.",
            "Margaret Thatcher was the first female Prime Minister of the United Kingdom."
        ]
        self.model_path = "/home/lab-admin/bert-base-cased-squad2" 
        self.tokenizer = None
        self.inputs = None
        self.selected_prompts = None
        self.selected_texts = None
        
    def get_id(self):
        return f"{self._model_name}-{self._batch_size}"

    def load_model(self):
        self._model = BertForQuestionAnswering.from_pretrained(
            self.model_path
        ).to(self._device)
    
    def load_data(self):
        self.selected_prompts = random.sample(self.prompts, self._batch_size)
        self.selected_texts = [self.contexts[self.prompts.index(prompt)] for prompt in self.selected_prompts]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.inputs = self.tokenizer(self.selected_prompts, self.selected_texts, padding=True, truncation=True, return_tensors="pt")

    def infer(self):
        self.inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model(**self.inputs)

        # Process outputs for each question-text pair
        total_tokens_generated = 0
        for i, (prompt, text) in enumerate(zip(self.selected_prompts, self.selected_texts)):
            answer_start_index = outputs.start_logits[i].argmax()
            answer_end_index = outputs.end_logits[i].argmax()
            predict_answer_tokens = self.inputs.input_ids[i, answer_start_index:answer_end_index + 1]
            num_tokens_generated = len(predict_answer_tokens)
            total_tokens_generated += num_tokens_generated
        return total_tokens_generated
    
    

def get_batched_inference_object(model_type, device_id, model, batch_size):
    if model_type == "vision":
        return VisionBatchedInference(device_id, model, batch_size)
    elif model_type == "llama":
        return LlamaBatchedInference(device_id, model, batch_size)
    elif model_type == "bert":
        return BertBatchedInference(device_id, model, batch_size)
    else:
        print("Unknown model-type. Must be one of 'vision', 'llama', 'bert'")
        sys.exit(1)
