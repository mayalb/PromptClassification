import streamlit as st
from langchain_community.llms import Ollama
import random
import math
import re

# --- Part 1: Generate Functions Dynamically ---
def extract_functions_from_chunks(chunks):
    # Step 1: Concatenate all chunks into a single string
    full_response = ''.join(chunks)

    # Step 2: Regular expression to extract function definitions (starting with 'def' and capturing until the function body is complete)
    function_pattern = r'def [\w_]+\(.*\):.*(?:\n(?: {4}.*\n?)*)*'
    functions = re.findall(function_pattern, full_response)

    # Clean up functions if necessary
    functions = [func.strip() for func in functions]  # Remove unnecessary whitespace

    return functions


def generate_functions(prompt, model="llama3", count=10):
    llm = Ollama(model=model)
    examples = []
    example_count = 0

    st.write("Sending prompt to LLM...")
    st.code(prompt)  # Display the prompt being sent

    try:
        chunks = []
        for chunk in llm.stream(prompt):
            chunks.append(chunk)  # Collect chunks in a list

        # Step 1: Extract the functions from the received chunks
        functions = extract_functions_from_chunks(chunks)
        st.write(f"Total functions generated: {len(functions)}")

        # Step 2: Check if no functions were generated
        if len(functions) == 0:
            st.error("No functions were generated. Please check your prompt or try again.")
            return []

        # Step 3: Try evaluating each function and only keep valid ones
        valid_functions = []

        for func in functions:
            try:
                # Check if the function has valid syntax by executing it in a safe environment
                # We will use `exec()` with a specific local context
                local_context = {}
                exec(func, {}, local_context)  # If invalid, this will raise an error
                valid_functions.append(func)  # Only append if no errors occurred
            except Exception as e:
                # Ignore any function that raises an error (invalid syntax or runtime)
                st.warning(f"Skipping invalid function due to error: {e}")
                continue  # Skip to the next function if error occurs

        if not valid_functions:
            st.error("No valid functions were generated after evaluation.")
            return []

        # Return the valid functions
        return valid_functions

    except Exception as e:
        st.error(f"Error while communicating with the LLM: {e}")
        return []
def evaluate_generated_functions(generated_function_strings):
    """ Convert function strings to actual callable functions. """
    generated_functions = []

    for func_str in generated_function_strings:
        try:
            # Use exec to dynamically define the function
            local_namespace = {}
            exec(func_str, {}, local_namespace)

            # Get the function from the local_namespace (assuming it is the only function in the string)
            function_name = list(local_namespace.keys())[0]
            generated_functions.append(local_namespace[function_name])
        except Exception as e:
            print(f"Error evaluating function: {e}")
            continue  # Skip invalid functions

    return generated_functions


# --- Part 3: Streamlit Interface ---
# --- Part 3: Streamlit Interface ---
# --- Part 3: Streamlit Interface ---
st.title("Dynamic Function Generation and Clustering")

# User input for the prompt
user_prompt = st.text_area("Enter your prompt to generate functions:", 
                           placeholder="Write your prompt here...")

if st.button("Generate Functions"):
    if not user_prompt.strip():
        st.error("Please enter a valid prompt.")
    else:
        # Generate functions based on the user's prompt using the llama_choose_correct_output
        st.write("Generating functions...")

        # Assuming 'generate_functions_from_prompt' is the function that uses LLaMA to generate functions
        generated_function_strings = generate_functions(user_prompt)

        # Check if functions are generated
        if not generated_function_strings:
            st.error("No functions were generated. Please check your prompt or try again.")
        else:
            # Evaluate the generated function strings into callable functions
            generated_functions = evaluate_generated_functions(generated_function_strings)
             # Check if callable functions are generated
            if not generated_functions:
                st.error("Failed to generate valid callable functions.")
            else:
                # Display the generated functions
                st.session_state.final_generated_functions = generated_functions
                st.write("### Generated Functions:")
                for i, func in enumerate(generated_function_strings, start=1):
                    st.subheader(f"Function {i}")
                    st.code(func, language="python")
