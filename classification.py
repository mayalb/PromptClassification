import streamlit as st
import pandas as pd
from transformers import pipeline
import io

# Initialize the zero-shot classification model with roberta-large-mnli
device = 0  # Set to 'cuda' if GPU is available, else 'cpu' for CPU
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", device=device)

# Define labels for classification
labels = [
    "Fully Detailed and Clear",
    "Clear but Missing Details",
    "Vague but Detailed",
    "Vague and Lacking Details",
]

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
        
        # Initialize the 'auto classification' column
        data['auto classification'] = None  # Start with None (empty) for all rows
        
        for index, row in data.iterrows():
            # Check if the "Prompt" column is not empty
            prompt = row["Prompt"]
            
            if pd.isna(prompt) or not str(prompt).strip():  # Skip empty or NaN prompts
                continue  # Do not update the 'auto classification' column for this row
            
            context = """
            Classify the following Python prompt based on clarity and completeness:
            - Fully Detailed and Clear: The prompt provides all necessary details, is specific, and directly addresses the task. Example: 'Write a Python function that sorts a list of integers in ascending order.'
            - Clear but Missing Details: The prompt is understandable but lacks some important details that would clarify the task further. Example: 'Write a function to sort a list.' (Doesn't specify the sorting order.)
            - Vague but Detailed: The prompt includes some specific details, but the overall meaning or context is unclear or ambiguous. Example: 'Sort this list based on some criteria.' (Lacks clarity about the criteria.)
            - Vague and Lacking Details: The prompt is both unclear and lacks important specifics. It lacks context and necessary details to complete the task effectively. Example: 'Sort the list.' (No information about the list, the type of sorting, or any criteria to guide the sorting process.)
            """
            
            full_prompt = context + str(prompt)  # Ensure the prompt is converted to string

            # Classify the prompt
            result = classifier(full_prompt, candidate_labels=labels)
            label = result['labels'][0]
            score = result['scores'][0]
            
            # Adjust threshold to ensure higher confidence for classification
            # if score < 0.4:  # If confidence is less than 75%, classify as "Vague and Lacking Details"
            #     label = "Vague and Lacking Details"
            
            # Assign the classification result to the 'auto classification' column
            data.at[index, 'auto classification'] = label

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
