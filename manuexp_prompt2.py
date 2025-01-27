import random
import math
from langchain_community.llms import Ollama
import streamlit as st
import openai
import os


# Define the sample functions for testing
def change_base_1(x, base):
    return ''.join(str(int(digit, 10)) for digit in format(x, 'b').zfill(base))

def change_base_2(x, base):
    return bin(x)[2:].zfill(base)

def change_base_3(x, base):
    if x < base:
        return str(x)
    else:
        return change_base_3(x // base, base) + str(x % base)

def change_base_4(x, base):
    return ''.join([str((x // (base ** i)) % base) for i in range(0, len(str(x)))])

def change_base_5(x, base):
    d = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    return ''.join([str((x // (base ** i)) % base) if (x // (base ** i)) % base < 10 else d[x // (base ** i) % 16] for i in range(0, len(str(x)))])

change_base_6 = lambda x, base: ''.join([str((x // (base ** i)) % base) if (x // (base ** i)) % base < 10 else chr(ord('a') + (x // (base ** i)) % 16 - 10) for i in range(0, len(str(x)))])

def change_base_7(x, base):
    return ''.join(map(lambda y: str(y) if y < 10 else chr(ord('a') + y - 10), [int(digit) for digit in format(x, 'b').zfill(base)]))

def change_base_8(x, base):
    d = {i: str(i) if i < 10 else chr(ord('a') + i - 10) for i in range(10)}
    return ''.join([d[int(digit)] for digit in format(x, 'b').zfill(base)])

def change_base_9(x, base):
    for i in range(len(str(x))):
        yield str((x // (base ** i)) % base) if (x // (base ** i)) % base < 10 else chr(ord('a') + (x // (base ** i)) % 16 - 10)

def change_base_10(x, base):
    result = ''
    for i in range(len(str(x))):
        digit = (x // (base ** i)) % base
        if digit < 10:
            result += str(digit)
        else:
            result += chr(ord('a') + digit - 10)
    return result

samples = [
    change_base_1,
    change_base_2,
    change_base_3,
    change_base_4,
    change_base_5,
    change_base_6,
    change_base_7,
    change_base_8,
    change_base_9,
    change_base_10
]



def llama_choose_correct_output(prompt):
    llm = Ollama(model="llama3")
    print("LLaMA Prompt Sent:\n", prompt)
    
    # Stream response
    response = ""
    try:
        for chunk in llm.stream(prompt):
            print(chunk, end="")
            response += chunk
    except Exception as e:
        print(f"Error communicating with LLaMA: {e}")
        return None  # Handle LLaMA errors gracefully

    print("\nLLaMA Response Received:\n", response.strip())
    return response.strip()


# Calculate entropy for a distinguishing input
def calculate_entropy(outputs):
    clusters = {}
    for i, output in enumerate(outputs):
        if output not in clusters:
            clusters[output] = []
        clusters[output].append(i)
    
    N = len(outputs)
    entropy = 0
    for cluster in clusters.values():
        nk = len(cluster)
        pk = nk / N
        entropy -= pk * math.log2(pk)
    return entropy, clusters


# Find distinguishing inputs and calculate entropies
def find_distinguishing_inputs_and_clusters_with_entropy(num_tests):
    distinguishing_inputs = []
    input_entropies = []
    best_inputs = []
    highest_entropy = -1
    best_clusters = None

    for _ in range(num_tests):
        x = random.randint(0, 100)
        base = random.randint(2, 10)
        outputs = [func(x, base) if func else None for func in samples]

        if len(set(outputs)) > 1:
            distinguishing_inputs.append((x, base))
            entropy, clusters = calculate_entropy(outputs)
            input_entropies.append((x, base, entropy, clusters))

            # Select the highest entropy input and break after the first
            if entropy > highest_entropy:
                highest_entropy = entropy
                best_inputs = [(x, base)]  # Start fresh list with this input
                best_clusters = clusters  # Update clusters
            elif entropy == highest_entropy:
                best_inputs.append((x, base))  # Add if entropy is the same

    # Return only the first best input with the highest entropy
    if best_inputs:
        return distinguishing_inputs, input_entropies, best_inputs, highest_entropy, best_clusters
    return distinguishing_inputs, input_entropies, None, highest_entropy, None


# Compare functions for equivalence
def compare_function_equivalence(outputs_list):
    function_count = len(outputs_list)
    clusters = []
    visited = [False] * function_count

    for i in range(function_count):
        if visited[i]:
            continue
        current_cluster = [i]
        visited[i] = True
        for j in range(i + 1, function_count):
            if not visited[j] and outputs_list[i] == outputs_list[j]:
                current_cluster.append(j)
                visited[j] = True
        clusters.append(current_cluster)

    return clusters


# Generate LLaMA Prompt using the proposed template
def generate_llama_prompt(x, base, clusters, outputs):
    prompt = f"""
# Specification selection as a multi-choice question  
Please select the correct output from a few choices for the function described below.

## Function signature and problem description:
def change-base(x, base):
    \"\"\" Function to change the base of a given number `x`. \"\"\"

## Multi-choice question to select the correct output:
What should be the value of `change_base(x = {x}, base = {base})`? Choose the correct option from the following:

"""
    # Add multi-choice options
    for idx, output in enumerate(set(outputs), start=1):
        prompt += f"- `Option {idx}`: {output}\n"

    # Add explanation request
    prompt += """
## Explanation:
Please explain your choice step by step and provide your final answer.
"""
    return prompt


# Streamlit UI
st.title("Function Clustering with Highest Entropy Distinguishing Inputs")

num_tests = st.sidebar.slider("Number of Tests", 1, 200, 50)

if st.button("Run Analysis"):
    st.write("Running analysis...")
    distinguishing_inputs, input_entropies, best_input, highest_entropy, best_clusters = find_distinguishing_inputs_and_clusters_with_entropy(num_tests)

    st.write("### Distinguishing Inputs")
    if distinguishing_inputs:
        for x, base in distinguishing_inputs:
            st.write(f"- x = {x}, base = {base}")
        st.write("total distinguishing inputs: ", len(distinguishing_inputs))
    else:
        st.write("No distinguishing inputs found.")

    # st.write("### Input Entropies")
    # for x, base, entropy, _ in input_entropies:
    #     st.write(f"- Input: x = {x}, base = {base}, Entropy = {entropy:.4f}")

    st.write("### Best Distinguishing Input (Highest Entropy)")
    if best_input:
        st.write(f"Highest Entropy: {highest_entropy:.4f}")
        for i, (x, base) in enumerate(best_input, start=1):
          st.write(f"- Selected Input {i}: x = {x}, base = {base}")

        # Calculate outputs for all functions using the selected input
        outputs = [func(x, base) if func else None for func in samples]
        st.write("### Outputs for All Functions")
        for i, output in enumerate(outputs, start=1):
            st.write(f"Function S{i}: {output}")

        # Cluster functions based on equivalence
        clusters = compare_function_equivalence(outputs)
        st.write("### Function Clusters (Equivalence)")
        for cluster_num, cluster in enumerate(clusters, start=1):
            st.write(f"Cluster {cluster_num}: Functions {', '.join([f'S{i+1}' for i in cluster])}")

        # Generate and send LLaMA prompt
        llama_prompt = generate_llama_prompt(x, base, clusters, outputs)
        st.write("###  Prompt")
        st.code(llama_prompt)

        # Get LLaMA response
        correct_output = llama_choose_correct_output(llama_prompt)
        st.write("### LLaMA Selected Output")
        st.write(correct_output)

        # correct_out = gpt4_choose_correct_output(llama_prompt)
        st.write("### GPT-4 Selected Output")
    

    else:
        st.write("No distinguishing input with high entropy found.")
