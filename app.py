
import os
from langchain_community.llms import Ollama


llm = Ollama(model="llama3")
prompt = """ Generate 10 implementations in Python for the function:

def change_base(x, base):
    Convert the number `x` to the specified `base` ( base must be less than 10) and return the result as a string.

Additional requirements:
- Ensure the function handles the case where `x = 0` by returning `"0"`.
- If the base is greater than or equal to 10, raise a `ValueError`.
- x should be pofitif, if not raise a `ValueError`.
"""

example_count = 0
examples = []


# Stream the output from the LLM
for chunks in llm.stream(prompt):
    # Print each chunk as it comes in
    print(chunks, end="")
    
    if "def " in chunks:  # Assuming code examples start with "def"
        examples.append(chunks)
        example_count += 1
    
    # Stop if we have 10 examples
    if example_count >= 10:
        break

# https://www.datacamp.com/tutorial/llama3-fine-tuning-locally?dc_referrer=https%3A%2F%2Fwww.google.com%2F