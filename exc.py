import streamlit as st
from langchain_community.llms import Ollama
import re
import random
import math
import string
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt

# --- Part 1: Generate Functions Dynamically ---
def extract_functions_from_chunks(chunks):
    full_response = ''.join(chunks)
    function_pattern = r'def [\w_]+\(.*\):.*(?:\n(?: {4}.*\n?)*)*'
    functions = re.findall(function_pattern, full_response)
    functions = [func.strip() for func in functions]  # Clean up the functions
    return functions

def generate_functions(prompt, model="llama3", count=10):
    llm = Ollama(model=model)
    chunks = []
    st.write("Sending prompt to LLM...")
    st.code(prompt)  # Display prompt

    try:
        for chunk in llm.stream(prompt):
            chunks.append(chunk)  # Collect chunks

        functions = extract_functions_from_chunks(chunks)
        st.write(f"Total functions generated: {len(functions)}")

        if len(functions) == 0:
            st.error("No functions were generated.")
            return []

        valid_functions = []
        for func in functions:
            try:
                local_context = {}
                exec(func, {}, local_context)  # Validate function syntax
                valid_functions.append(func)
            except Exception as e:
                st.warning(f"Skipping invalid function: {e}")
                continue
        if not valid_functions:
            st.error("No valid functions were generated.")
            return []
        return valid_functions

    except Exception as e:
        st.error(f"Error while communicating with LLaMA: {e}")
        return []
    
def change_base(x, base):
    """Function to change the base of a given number x."""
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36.")
    if x < 0:
        raise ValueError("x must be non-negative.")
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    while x > 0:
        result = digits[x % base] + result
        x //= base
    return result if result else "0"

# Function to evaluate function strings and return callable functions
import inspect

def evaluate_generated_functions(function_inputs):
    """
    Evaluate and filter valid functions from the provided list of function strings or objects.
    Functions that raise any runtime errors (wrong input, invalid names, etc.) will be ignored.
    """
    evaluated_functions = []

    for func_input in function_inputs:
        try:
            # If the input is a string (e.g., code), we need to handle it
            if isinstance(func_input, str):
                func_str = func_input
                # Dynamically execute the function source
                exec_globals = {}
                exec(func_str, exec_globals)

                # Retrieve the function object by searching the global namespace
                func = next(
                    (f for f in exec_globals.values() if callable(f) and f.__name__ == "change_base"),
                    None,
                )
            elif callable(func_input):
                # If it's already a callable function object, we can skip code extraction
                func = func_input
            else:
                print(f"Invalid input: Expected string or function but got {type(func_input)}")
                continue

            # Test the function with sample inputs to check if it executes correctly
            try:
                # Test with dummy inputs to check if the function works
                func(10, 2)  # Example: Convert number 10 to base 2 (adjust as necessary)
                evaluated_functions.append(func)  # Add valid function to the list
            except Exception as test_error:
                # If the function raises any error (wrong input, invalid function call, etc.), ignore it
                print(f"Function {func.__name__} skipped due to runtime error: {test_error}")
        except Exception as exec_error:
            # Log any code execution error and skip the function
            print(f"Function skipped due to error: {exec_error}")
            continue

    return evaluated_functions


# --- Part 2: Function Clustering and Analysis ---
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


def filter_change_base_functions(generated_functions):
    # Keep only functions that have the '__name__' attribute and the name 'change_base'
    valid_functions = []
    for func in generated_functions:
        if hasattr(func, '__name__'):  # Check if the function has the '__name__' attribute
            if func.__name__ == 'change_base':  # Check if the function's name is 'change_base'
                valid_functions.append(func)  # Add the function to the list
    return valid_functions




def find_distinguishing_inputs_and_clusters_with_entropy(num_tests, generated_function_strings):
    # Evaluate the function strings into callable functions
    generated_functions = evaluate_generated_functions(generated_function_strings)

    distinguishing_inputs = []
    input_entropies = []
    best_inputs = []
    highest_entropy = -1
    best_clusters = None

    for _ in range(num_tests):
        x = random.randint(0, 100)
        base = random.randint(2, 10)
        outputs = [func(x, base) if func else None for func in generated_functions]

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
## Return:
Please return the result like this: My final answer is: Option x.
"""
    return prompt

import re

def extract_final_answer(llm_response):
    """
    Extracts the final answer in the format 'Option X' from the LLM response.
    """
    match = re.search(r"My final answer is: Option (\d+)", llm_response)
    if match:
        return f"Option {match.group(1)}"
    return None  # Return None if no valid match is found

# def extract_final_answer(llama_response):
#     if not llama_response:  # Handle None or empty response
#         return None  

#     # Match patterns like "Option 1: 33" and extract only the number
#     match = re.search(r'Option\s*\d+:\s*([\d.]+)', llama_response)
#     if match:
#         return match.group(1).strip()  # Extract only the number (e.g., "33")

#     # If "Option X:" is not present, try other patterns
#     patterns = [
#         r'Final Answer:\s*(?:The final answer is\s*)?([^\.\n]+)',
#         r'The final answer is:\s*([^\.\n]+)',
#         r'The correct answer is:\s*([^\.\n]+)',
#         r'So, the correct answer is\s*([^\.\n]+)',
#         r'The correct output is:\s*([^\.\n]+)',
#         r'The correct choice is:\s*([^\.\n]+)',
#         r'I would select:\s*([^\.\n]+)',
#         r'I would choose:\s*([^\.\n]+)',
#         r'The correct option should be:\s*([^\.\n]+)',
#         r'My answer is:\s*([^\.\n]+)',
#         r'Our final answer is:\s*([^\.\n]+)',
#         r'My final answer is:\s*([^\.\n]+)'
#     ]

#     for pattern in patterns:
#         match = re.search(pattern, llama_response, re.IGNORECASE)
#         if match:
#             return match.group(1).strip()  # Extract and return clean answer

#     return None  # If no match found, return None

def compare_with_correct_output(outputs, correct_output):
    # Debugging: Print outputs and correct_output to check if they're in the expected format
    st.write(f"Outputs: {outputs}")
    st.write(f"Correct Output: {correct_output}")

    matching_functions = []
    for i, output in enumerate(outputs, start=1):
        # Clean both output and correct_output before comparing
        if str(output).strip().lower() == str(correct_output).strip().lower():  # Allow case-insensitive comparison
            matching_functions.append(f"Function S{i}")

    return matching_functions
# --- Streamlit Interface ---
st.title("Dynamic Function Generation and Clustering")

# Create tabs for different interfaces
tab1, tab2, tab3, tab4 = st.tabs(["Generate Functions","Test functions", "Cluster the functions","Statistics"])

# --- Tab 1: Generate Functions ---
with tab1:
    # User input for the prompt
    user_prompt = st.text_area("Enter your prompt to generate functions:", placeholder="Write your prompt here...")

    if st.button("Generate Functions"):
        if not user_prompt.strip():
            st.error("Please enter a valid prompt.")
        else:
            st.write("Generating functions...")

            # Generate functions from the prompt
            generated_function_strings = generate_functions(user_prompt)

            if not generated_function_strings:
                st.error("No functions generated.")
            else:
                # Evaluate the generated functions
                generated_functions = evaluate_generated_functions(generated_function_strings)

                if not generated_functions:
                    st.error("Failed to generate valid callable functions.")
                else:
                    # Store the final valid functions in session state
                    st.session_state.final_generated_functions = generated_functions # Store in session state
                    st.write("Generated Functions Code:")

                    st.write("### Generated Functions:")
                    for i, func in enumerate(generated_function_strings, start=1):
                        st.subheader(f"Function {i}")
                        st.code(func, language="python")

# --- Tab 2: Analyze Functions ---
with tab2:
    if 'final_generated_functions' in st.session_state:
        st.write("### Test Generated Functions with Fixed Inputs")
        fixed_inputs = st.text_area("Enter test cases (comma-separated x, base pairs):", "5, 9\n45, 7\n87, 5")
        generated_functions = evaluate_generated_functions(st.session_state.final_generated_functions)

        if st.button("Run Tests"):
            try:
                # Parse the input
                test_cases = []
                for line in fixed_inputs.strip().split("\n"):
                    x, base = map(int, line.split(","))
                    test_cases.append((x, base))

                # Run tests
                st.write("### Test Results")
                for x, base in test_cases:
                    st.subheader(f"Test Case: x = {x}, base = {base}")
                    results = []
                    for i, func in enumerate(generated_functions, start=1):
                        try:
                            result = func(x, base)
                            results.append((f"Function {i}", result))
                        except Exception as e:
                            results.append((f"Function {i}", f"Error: {e}"))
                    
                    # Display results for the current test case
                    for func_name, output in results:
                        st.write(f"{func_name} → Output: {output}")

            except Exception as e:
                st.error(f"Error parsing inputs or testing functions: {e}")
    else:
        st.info("Please generate functions first in the Generate Functions tab.")

# --- Tab 3: Analyze Functions ---
import traceback

with tab3:
    # Check if generated functions exist in session state
    if 'final_generated_functions' in st.session_state:
        generated_functions = st.session_state.final_generated_functions  # Retrieve from session state
        
        # Filter out only valid functions (those that are not causing errors)
        valid_functions = []
        for func in generated_functions:
            try:
                # Try to run a simple test (e.g., evaluate it on a basic input)
                test_input = (1, 10)  # Replace with valid inputs for your functions
                func(*test_input)  # Run the function with the test input
                valid_functions.append(func)  # If no error occurs, add to valid list
            except Exception as e:
                st.write(f"Invalid function {func} due to error: {e}")

        if not valid_functions:
            st.error("No valid functions found. Please check the generated functions.")
        else:
            # Proceed with the analysis
            tests = 10  # Set the number of automated tests
            correct_matches = 0  # Track correct matches
            num_tests = st.sidebar.slider("Number of Tests", 1, 200, 50)

            if st.button("Run Analysis"):
                st.session_state.run_analysis = True  # Store in session state to persist

            if st.session_state.get("run_analysis", False):  # Check if analysis was run
                st.write("Running analysis...")
                for test in range(tests):
                    # Perform the analysis with only valid functions
                    st.write(f"### Running Test {test + 1}/{tests}")
                    distinguishing_inputs, input_entropies, best_inputs, highest_entropy, best_clusters = find_distinguishing_inputs_and_clusters_with_entropy(num_tests, valid_functions)

                    # Display Distinguishing Inputs
                    st.write("### Best Distinguishing Input (Highest Entropy)")
                    if best_inputs:
                        if isinstance(best_inputs, list) and all(isinstance(item, tuple) and len(item) == 2 for item in best_inputs):
                            st.write(f"Highest Entropy: {highest_entropy:.4f}")
                            for i, (x, base) in enumerate(best_inputs, start=1):
                                st.write(f"- Selected Input {i}: x = {x}, base = {base}")

                        # Calculate outputs for all functions using the selected input
                        outputs = [func(x, base) if func else None for func in valid_functions]
                        st.write("### Outputs for All Functions")
                        for i, output in enumerate(outputs, start=1):
                            st.write(f"Function S{i}: {output}")

                        # Cluster functions based on equivalence
                        clusters = compare_function_equivalence(outputs)
                        st.write("### Function Clusters (Equivalence)")
                        for cluster_num, cluster in enumerate(clusters, start=1):
                            st.write(f"Cluster {cluster_num}: Functions {', '.join([f'S{i+1}' for i in cluster])}")

                        # --- Allow User to Select Correct Cluster ---
                        if test==0:

                            st.write("### Select the Correct Cluster")

                            # Create dropdown options based on identified clusters
                            cluster_options = [f"Cluster {i+1}" for i in range(len(clusters))]

                            # Maintain user selection in session state
                            if "correct_cluster" not in st.session_state:
                                st.session_state.correct_cluster = cluster_options[0]  # Default to first cluster

                            # Dropdown for selecting the correct cluster
                            selected_cluster = st.selectbox("Select the correct cluster:", cluster_options, index=cluster_options.index(st.session_state.correct_cluster))

                            if st.button("Confirm Selection"):
                                st.session_state.cluster_selected = True  # Store selection flag
                                st.session_state.correct_cluster = selected_cluster  # Save selected cluster

                        if st.session_state.get("cluster_selected", False):
                            st.write(f"✅ You selected: {st.session_state.correct_cluster}")

                            # Generate and send LLaMA prompt
                            llama_prompt = generate_llama_prompt(x, base, clusters, outputs)
                            # st.write("### Prompt")
                            # st.code(llama_prompt)

                            # Get LLaMA response
                            correct_output = llama_choose_correct_output(llama_prompt)
                            # st.write("### LLaMA Selected Output")
                            # st.write(correct_output)

                            # --- Treatment: Map Selected Output to Functions and Clusters ---
                            st.write("### Treatment: Mapping Selected Output to Functions and Clusters")

                            if correct_output is None:
                                correct_output = ""
                            correct_output = extract_final_answer(correct_output)
                            if not correct_output or not correct_output.split():
                                st.error("⚠️ Could not extract a valid answer from LLaMA's response.")
                                correct_output = None
                            else:
                                try:
                                    option_number = int(correct_output.split()[-1])  # Get the number after "Option"
                                    correct_output = list(set(outputs))[option_number - 1]  # Get corresponding value
                                except (ValueError, IndexError):
                                    st.write("Error: Unable to determine the correct output from LLaMA's response.")
                                    correct_output = None

                            matching_functions = compare_with_correct_output(outputs, correct_output)
                            st.write(f"Matching Functions: {matching_functions}")

                            # Identify clusters that contain matching functions
                            llm_selected_cluster = []
                            for cluster_num, cluster in enumerate(clusters, start=1):
                                for func_index in cluster:
                                    if outputs[func_index] == correct_output:
                                        llm_selected_cluster.append(f"Cluster {cluster_num}")
                                        break  

                            # Display the results
                            st.write("#### Functions Matching the Selected Output:")
                            st.write(", ".join(matching_functions) if matching_functions else "No functions match the selected output.")
                            st.write("#### Clusters Containing Matching Functions:")
                            st.write(", ".join(llm_selected_cluster) if llm_selected_cluster else "No clusters contain functions that match the selected output.")
                            st.write("### Comparison of User Selection and LLM Prediction")
                            if "selected_cluster" not in st.session_state:
                              st.session_state.selected_cluster = None
                            if llm_selected_cluster:
                                st.write(f"🤖 LLM Selected Cluster: {llm_selected_cluster}")
                                st.write(f"👤 User Selected Cluster: {st.session_state.correct_cluster}")


                                if st.session_state.correct_cluster in llm_selected_cluster:
                                    st.success("✅ The LLM's choice matches your selection!")
                                    correct_matches += 1
                                else:
                                    st.error("❌ The LLM's choice is different from your selection.")
                            else:
                                st.write("⚠️ LLM did not select a valid cluster.")

                    else:
                        st.write("No distinguishing input with high entropy found.")
            accuracy = (correct_matches / tests) * 100
            st.write(f"### Final Accuracy: {accuracy:.2f}% ({correct_matches}/{tests} correct matches)")
    else:
        st.info("Please generate functions first in the Generate Functions tab.")

with tab4:
    
    st.title("Clustering Evaluation: Number of Clusters vs. Test Results")

    # File upload for file1
    uploaded_file1 = st.file_uploader("Upload your first Excel file (file1)", type=["xlsx"])

    # File upload for file2
    uploaded_file2 = st.file_uploader("Upload your second Excel file (file2)", type=["xlsx"])

    if uploaded_file1 is not None:
        # Read the first Excel file using pandas
        df1 = pd.read_excel(uploaded_file1)

        # Display the first few rows of the dataframe for file1
        st.write("Data Preview from File 1:", df1.head())

        # Ensure the 'cluster' and 'result' columns are present in file1
        if 'cluster' in df1.columns and 'result' in df1.columns:
            # Convert columns to numeric (if they aren't already)
            df1['cluster'] = pd.to_numeric(df1['cluster'], errors='coerce')
            df1['result'] = pd.to_numeric(df1['result'], errors='coerce')

            # Drop rows with NaN values (if any invalid data exists in either column)
            df1 = df1.dropna(subset=['cluster', 'result'])

            # Filter to only include clusters between 1 and 10 (inclusive), and round to integers
            df1 = df1[(df1['cluster'] >= 1) & (df1['cluster'] <= 10)]
            df1['cluster'] = df1['cluster'].astype(int)

            # Sort data by the 'cluster' column (to ensure the line is drawn correctly)
            df1 = df1.sort_values(by='cluster')

            # Extract the relevant columns from file1
            cluster_data1 = df1['cluster']
            result_data1 = df1['result']

            # Plot the graph with a line for the first file (file1)
            plt.figure(figsize=(10, 6))
            plt.plot(cluster_data1, result_data1, color='b', label="File1: Intra prompts")  # Line plot without markers
            plt.title("Number of Clusters vs. Test Results")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Test Results")
            plt.xticks(range(1, 11))  # Set x-axis ticks to integer values between 1 and 10
            plt.grid(True)

            # Process the second file (file2) and plot the lines
            if uploaded_file2 is not None:
                # Read the second Excel file (file2)
                df2 = pd.read_excel(uploaded_file2)

                # Display the first few rows of the dataframe for file2
                st.write("Data Preview from File 2:", df2.head())

                # Ensure the 'cluster' and 'result' columns are present in file2
                if 'cluster' in df2.columns and 'result' in df2.columns:
                    # Convert columns to numeric for file2
                    df2['cluster'] = pd.to_numeric(df2['cluster'], errors='coerce')
                    df2['result'] = pd.to_numeric(df2['result'], errors='coerce')

                    # Drop rows with NaN values (if any)
                    df2 = df2.dropna(subset=['cluster', 'result'])

                    # Filter clusters between 1 and 10 (inclusive) and round to integers
                    df2 = df2[(df2['cluster'] >= 1) & (df2['cluster'] <= 10)]
                    df2['cluster'] = df2['cluster'].astype(int)

                    # Sort data by the 'cluster' column
                    df2 = df2.sort_values(by='cluster')

                    # Extract the relevant columns from file2
                    cluster_data2 = df2['cluster']
                    result_data2 = df2['result']

                    # Plot the data from file2 as a line (e.g., in orange)
                    plt.plot(cluster_data2, result_data2, color='orange', label="File2: Inter prompts")

                    # Add a legend for the lines
                    plt.legend()

                    # Show the plot in Streamlit
                    st.pyplot(plt)
                else:
                    st.error("File2 must contain 'cluster' and 'result' columns.")
            else:
                st.info("Please upload the second Excel file (file2).")
        else:
            st.error("The first file (file1) must contain 'cluster' and 'result' columns.")
    else:
        st.info("Please upload the first Excel file (file1).")