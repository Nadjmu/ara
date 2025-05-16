import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

# Title
st.title("Multi-File Correlation Heatmap")

# List of Excel files to load
file_names = ["step_0.xlsx", "step_1.xlsx", "step_2.xlsx", "step_3.xlsx", "step_4.xlsx","boundary.xlsx"]

# Load all DataFrames into a dictionary
dfs = {}
for file in file_names:
    try:
        dfs[file] = pd.read_excel(file)
    except FileNotFoundError:
        st.warning(f"File {file} not found. Skipping...")

if not dfs:
    st.error("No valid Excel files found. Please check file names.")
    st.stop()

# Configure date range picker with fixed boundaries
date_range = st.date_input(
    "📅 Select Date Range (1.1.2015 - 26.2.2025)",
    value=[date(2015, 1, 1), date(2025, 2, 26)],  # Default range set to full period
    min_value=date(2015, 1, 1),  # Earliest allowed date
    max_value=date(2025, 2, 26),  # Latest allowed date
    key="date_range"
)

# Display selection only when full range is selected
if len(date_range) != 2:
    st.warning("Please select both start and end dates")

# Function to convert a date to the corresponding index
def date_to_index(selected_date: date):
    base_date = date(2015, 1, 1)  # 12/31/2014 is index 1
    delta = selected_date - base_date
    return delta.days

# Function to convert a selected date range to index range
def convert_date_range_to_index(start_date: date, end_date: date):
    start_index = date_to_index(start_date)
    end_index = date_to_index(end_date)
    return [start_index, end_index]

#st.write("### Date Range to Index Conversion")
start_index, end_index = convert_date_range_to_index(date_range[0], date_range[1])
#st.write(f"Start Index: {start_index}, End Index: {end_index}")


# Allow column selection from each file
st.write("### Select Variables for Correlation")
selected_columns_dict = {}  # Dictionary to store selected columns for each file
column_mapping = {}  # To store original column names to prefixed names

for i, (file, df) in enumerate(dfs.items()):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        # Create prefixed column names for display
        prefixed_cols = [f"{i}_{col}" for col in numeric_cols]
        # Store mapping from prefixed to original names
        column_mapping.update({prefixed: original for prefixed, original in zip(prefixed_cols, numeric_cols)})
        
        cols = st.multiselect(
            f"Select columns from {file} (File {i}):",
            options=prefixed_cols,
            key=f"{file}_cols"  # Unique key for each multiselect
        )
        # Store original column names in the dictionary
        selected_columns_dict[file] = [column_mapping[col] for col in cols]
    else:
        st.warning(f"No numeric columns found in {file}.")

# Flatten all selected columns
all_selected_columns = [col for cols in selected_columns_dict.values() for col in cols]

if len(all_selected_columns) >= 2:
    # Combine selected columns from all DataFrames
    st.write(f"Selections: {len(all_selected_columns)}")
    
    # Create a list of DataFrames containing only the selected columns
    data_to_combine = []
    for i, (file, cols) in enumerate(selected_columns_dict.items()):
        if cols:  # Only include if columns were selected
            df = dfs[file][cols].copy()
            # Rename columns with prefix
            df.columns = [f"{i}_{col}" for col in df.columns]
            data_to_combine.append(df)
    
    # Combine all selected columns
    combined_data = pd.concat(data_to_combine, axis=1)
    filtered_data = combined_data.loc[
        (combined_data.index >= start_index) & (combined_data.index <= end_index)
    ]
    st.write("### Combined DataFrame (shows the first 5 rows)")
    st.dataframe(filtered_data.head())
    
    # Compute correlation matrix
    corr_matrix = filtered_data.corr()

    # Plot heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
        fmt=".2f",
        center=0,
        ax=ax
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig)
else:
    st.warning("Please select at least 2 numeric columns across all files.")