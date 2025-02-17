import streamlit as st
import pandas as pd
from transformers import pipeline
import io

# Initialize the zero-shot classification model with roberta-large-mnli
device = 0  # Set to 'cuda' if GPU is available, else 'cpu' for CPU
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=device)

# Define labels for classification
labels = [
    "The prompt is clear and complete",
    "The prompt is clear but incomplete",
    "The prompt is unclear but complete",
    "The prompt is unclear and incomplete"
]

# Classification context
# context = """
# Classify the following Python prompt based on clarity and completeness:
# - The prompt is clear and complete: It is well-structured, easy to understand, and contains all necessary requirements.
# - The prompt is clear but incomplete: It is understandable but lacks some necassery requirements.
# - The prompt is unclear but complete: It has the necessary details but is difficult to understand due to ambiguity.
# - The prompt is unclear and incomplete: It is ambiguous  and lacks essential information.
# """
# context = """
# Classify the following Python prompt based on clarity and completeness:

# - The prompt is **clear and complete**: It is easy to understand, and contains all necessary details for execution. 
#   Example: "Write a Python function that sorts a list of integers in ascending order and allows an optional parameter for descending order."

# - The prompt is **clear but incomplete**: It is understandable but lacks important requirements, making execution difficult. 
#   Example: "Write a function to sort a list." (It does not specify the sorting order or data type.)

# - The prompt is **unclear but complete**: It includes all necessary details but it's ambiguous and confusing. 
#   Example: "Sort numbers in some order." (It contains enough information but is vague.)

# - The prompt is **unclear and incomplete**: It is both hard to understand and missing critical requirements. 
#   Example: "Sort the list." (No mention of sorting order, data type, or any additional specifications.)

# Classify the following prompt accordingly:
# """
context = """
Determine whether the following Python prompt is clear and complete:

- **Clear and Complete (Yes/Yes)**: The prompt is easy to understand and includes all necessary requirements for execution.
  Example: "Write a Python function that sorts a list of integers in ascending order and allows an optional parameter for descending order."

- **Clear but Incomplete (Yes/No)**: The prompt is easy to understand, but lacks essential requirements needed for proper execution.
  Example: "Write a function to sort a list." (Does not specify the sorting order or data type.)

- **Unclear but Complete (No/Yes)**: The prompt includes all necessary requirements but is ambiguous and confusing.
  Example: "Sort numbers in some order." (Has enough information but is vague.)

- **Unclear and Incomplete (No/No)**: The prompt ambiguous and lacks critical requirements needed for execution.
  Example: "Sort the list." (No mention of sorting order, data type, or additional requirements.)

Classify the following prompt accordingly:
"""


# Streamlit app title
st.title("Prompt Classifier")

# File upload section
uploaded_file = st.file_uploader("Upload an Excel file with 'Prompt' column", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded Excel file
    data = pd.read_excel(uploaded_file)

    # Check if the required column 'Prompt' exists
    if "Prompt" in data.columns:
        st.write("Excel file uploaded successfully!")
        
        # Display the first few rows of the uploaded file
        st.write("Here are the first few rows of your file:")
        st.write(data.head())

        # Process prompts
        st.write("Processing prompts...")
        
        # Initialize the new classification columns
        data['Clear'] = None
        data['Complete'] = None
        
        for index, row in data.iterrows():
            prompt = row["Prompt"]
            
            if pd.isna(prompt) or not str(prompt).strip():  # Skip empty or NaN prompts
                continue  # Do not update classification columns for this row
            
            full_prompt = context + "\nPrompt: " + str(prompt)
            
            # Single classification call
            result = classifier(full_prompt, candidate_labels=labels)
            top_label = result['labels'][0]
            
            # Assign classification results based on the selected label
            if "clear and complete" in top_label:
                data.at[index, 'Clear'] = "Yes"
                data.at[index, 'Complete'] = "Yes"
            elif "clear but incomplete" in top_label:
                data.at[index, 'Clear'] = "Yes"
                data.at[index, 'Complete'] = "No"
            elif "unclear but complete" in top_label:
                data.at[index, 'Clear'] = "No"
                data.at[index, 'Complete'] = "Yes"
            else:
                data.at[index, 'Clear'] = "No"
                data.at[index, 'Complete'] = "No"

        # Save the modified DataFrame back to a BytesIO object
        output_filename = "classified_prompts.xlsx"
        output = io.BytesIO()
        data.to_excel(output, index=False, engine='openpyxl')  # Write the updated data to BytesIO object
        output.seek(0)  # Go to the start of the BytesIO buffer

        # Display the downloadable link
        st.success("Classification completed! Download your updated file below:")
        st.download_button(
            label="Download Excel with Classifications",
            data=output,
            file_name=output_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.error("The uploaded file does not contain a 'Prompt' column. Please upload a valid file.")
else:
    st.info("Please upload an Excel file to start the classification.")
