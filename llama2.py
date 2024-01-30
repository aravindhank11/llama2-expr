import torch
import sys
import timeit
from transformers import LlamaTokenizer, LlamaForCausalLM

# Global constants
DEVICE = torch.device('cuda:0')
MODEL_PATH = 'openlm-research/open_llama_7b_v2'
REPEAT = 2
OUTPUT_TOKENS = 500
PROMPTS = [
    "A short summary on the hottest researched topic",
    "What are the implications of AI on society",
    "Imagine a future with no human beings",
    "Discuss the current state of renewable energy"
]

# Global counters
total_time = 0
tokens_generated = 0


# Given model, input, and expected output size => use LLM to infer
def infer(model, input_token, max_new_tokens=500):
    generation_output = model.generate(
        input_ids=input_token, max_new_tokens=max_new_tokens
    )
    global tokens_generated
    tokens_generated += generation_output.size()[1]


# Tokenizer to convert string to input tokens
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)

# Load the model
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16 #, device_map='auto',
).to(DEVICE)

# Create input tokens from prompts
input_tokens = []
for prompt in PROMPTS:
    input_token = tokenizer(prompt, return_tensors="pt").input_ids
    input_token = input_token.to(DEVICE)
    input_tokens.append(input_token)


# Run Expr
for input_token in input_tokens:
    execution_time = timeit.timeit(
        stmt=lambda: infer(model, input_token, max_new_tokens=OUTPUT_TOKENS),
        number=REPEAT
    )
    total_time += execution_time
    print(".", end="")
    sys.stdout.flush()

# Print result
print("Generated {} tokens/second".format(tokens_generated / total_time))
