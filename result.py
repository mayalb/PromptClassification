import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit App
st.title("Clustering Evaluation: Number of Clusters vs. Test Results")

# File upload from the user
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file using pandas
    df = pd.read_excel(uploaded_file)
    
    # Display the first few rows of the dataframe to the user for verification
    st.write("Data Preview:", df.head())

    # Ensure the 'cluster' and 'result' columns are present
    if 'cluster' in df.columns and 'result' in df.columns:
        # Convert columns to numeric (if they aren't already)
        df['cluster'] = pd.to_numeric(df['cluster'], errors='coerce')
        df['result'] = pd.to_numeric(df['result'], errors='coerce')

        # Drop rows with NaN values (if any invalid data exists in either column)
        df = df.dropna(subset=['cluster', 'result'])

        # Filter to only include clusters between 1 and 10 (inclusive), and round to integers
        df = df[(df['cluster'] >= 1) & (df['cluster'] <= 10)]
        df['cluster'] = df['cluster'].astype(int)

        # Sort data by the 'cluster' column (to ensure the line is drawn correctly)
        df = df.sort_values(by='cluster')

        # Extract the relevant columns
        cluster_data = df['cluster']
        result_data = df['result']

        # Plot the graph with a line (instead of points) for Number of Clusters vs. Test Results
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_data, result_data, color='b', label="Test Results")  # Line plot without markers
        plt.title("Number of Clusters vs. Test Results")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Test Results")
        plt.xticks(range(1, 11))  # Set x-axis ticks to integer values between 1 and 10
        plt.grid(True)
        plt.legend()

        # Show the plot in Streamlit
        st.pyplot(plt)
    else:
        st.error("The Excel file must contain 'cluster' and 'result' columns.")
else:
    st.info("Please upload an Excel file containing 'cluster' and 'result' columns.")
