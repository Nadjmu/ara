import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(layout="wide")  # Must be the first Streamlit command

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

# Sidebar navigation
with st.sidebar:
    selected = st.radio(
        "",
        ["Descriptive Statistics", "Correlation"],
        index=0
    )

#####################CONTENT#####################
# Main content based on selection

if selected == "Descriptive Statistics":
    st.title("Descriptive Statistics")

    variable, diagram = st.columns([1,3])
    with variable:
        st.write("### Select Date Range")
        # Configure date range picker with fixed boundaries
        date_range = st.date_input(
            "📅 Select Date Range (1.1.2015 - 26.2.2025)",
            value=[date(2015, 1, 1), date(2025, 2, 26)],  # Default range set to full period
            min_value=date(2015, 1, 1),  # Earliest allowed date
            max_value=date(2025, 2, 26),  # Latest allowed date
            key="desc_date_range"  # Unique key for this tab
        )

        # Display selection only when full range is selected
        if len(date_range) != 2:
            st.warning("Please select both start and end dates")
        start_index, end_index = convert_date_range_to_index(date_range[0], date_range[1])
        
        # Allow column selection from each file
        st.write("### Select Variables")
        selected_columns_dict = {}  # Dictionary to store selected columns for each file
        for i, (file, df) in enumerate(dfs.items()):
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numeric_cols:
                cols = st.multiselect(
                    f"Select columns from {file} (File {i}):",
                    options=numeric_cols,  # Show original column names
                    key=f"desc_{file}_cols"  # Unique key for each multiselect
                )
                # Store original column names in the dictionary
                selected_columns_dict[file] = cols
            else:
                st.warning(f"No numeric columns found in {file}.")
                
    with diagram:
        # Find the first non-empty list of selected columns
        first_variable = None
        for columns in selected_columns_dict.values():
            if columns:  # Non-empty list
                first_variable = columns[0]  # Take the first variable
                break

        if first_variable:
            # --- 1. Data Preparation ---
            selected_data = pd.DataFrame()
            for i, (file, df) in enumerate(dfs.items()):
                if first_variable in df.columns:
                    filtered_df = df.loc[
                        (df.index >= start_index) & (df.index <= end_index),
                        [first_variable]
                    ]
                    filtered_df.columns = [f"{file}_{first_variable}"]  # Use filename instead of index
                    selected_data = pd.concat([selected_data, filtered_df], axis=1)

            # --- 2. Plot Customization ---
            st.write("### Enhanced Visualization")

            # Reset index for plotting
            plot_data = selected_data.reset_index()
            plot_data_melted = plot_data.melt(
                id_vars=plot_data.columns[0], 
                var_name='Source',  # More descriptive than 'Series'
                value_name=first_variable  # Use variable name as y-axis label
            )

            # --- 3. Seaborn Styling ---
            sns.set_style("whitegrid")  # Clean white background with grid
            sns.set_palette("husl")  # Use a vibrant color palette
            
            # Create figure with constrained layout
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            
            # Plot with improved aesthetics
            sns.lineplot(
                data=plot_data_melted,
                x=plot_data.columns[0],  # Time axis
                y=first_variable,
                hue='Source',
                style='Source',  # Different line styles
                markers=True,  # Show markers on lines
                dashes=False,  # Solid lines
                linewidth=2.5,
                ax=ax
            )

            # --- 4. Advanced Customization ---
            # Dynamic title with date range
            title = f"Trend of '{first_variable}' ({date_range[0]} to {date_range[1]})"
            ax.set_title(title, fontsize=16, pad=20, fontweight='bold')

            # Axis labels
            ax.set_xlabel("Date/Time", fontsize=12)
            ax.set_ylabel(first_variable, fontsize=12)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Customize grid
            ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
            ax.set_axisbelow(True)  # Grid behind data

            # Legend customization
            plt.legend(
                title='Data Source',
                bbox_to_anchor=(1.05, 1),  # Move legend outside plot
                loc='upper left',
                frameon=True,
                shadow=True
            )

            # Tight layout to prevent clipping
            plt.tight_layout()

            # --- 5. Display in Streamlit ---
            st.pyplot(fig)

elif selected == "Correlation":
    st.title("Correlation Heatmap")
    variable, diagram = st.columns([1,3])
    with variable:
        # Configure date range picker with fixed boundaries
        date_range = st.date_input(
            "📅 Select Date Range (1.1.2015 - 26.2.2025)",
            value=[date(2015, 1, 1), date(2025, 2, 26)],  # Default range set to full period
            min_value=date(2015, 1, 1),  # Earliest allowed date
            max_value=date(2025, 2, 26),  # Latest allowed date
            key="corr_date_range"  # Unique key for this tab
        )

        # Display selection only when full range is selected
        if len(date_range) != 2:
            st.warning("Please select both start and end dates")

        start_index, end_index = convert_date_range_to_index(date_range[0], date_range[1])

        # Allow column selection from each file
        st.write("### Select Variables for Correlation")
        selected_columns_dict = {}  # Dictionary to store selected columns for each file

        for i, (file, df) in enumerate(dfs.items()):
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numeric_cols:
                cols = st.multiselect(
                    f"Select columns from {file} (File {i}):",
                    options=numeric_cols,  # Show original column names
                    key=f"corr_{file}_cols"  # Unique key for each multiselect
                )
                # Store original column names in the dictionary
                selected_columns_dict[file] = cols
            else:
                st.warning(f"No numeric columns found in {file}.")
        # Flatten all selected columns
        all_selected_columns = [col for cols in selected_columns_dict.values() for col in cols]

        if len(all_selected_columns) >= 2:
            
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
        else:
            st.warning("Please select at least 2 numeric columns across all files.")

    with diagram:
            # Compute correlation matrix
            corr_matrix = filtered_data.corr()

            # Plot heatmap
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