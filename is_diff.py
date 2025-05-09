import pandas as pd

def compare_excel_columns(file_path):
    """
    Compare column 2 of 'BO4' sheet with column 4 of 'AO3' sheet in an Excel file.
    
    Returns:
    - A tuple containing (are_identical, differences)
      where are_identical is a boolean and differences is a DataFrame showing discrepancies
    """
    # Read both worksheets
    try:
        bo4_df = pd.read_excel(file_path, sheet_name=10)
        ao3_df = pd.read_excel(file_path, sheet_name=3)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None
    
    # Extract the columns to compare
    # Note: pandas uses 0-based indexing for columns, so we adjust accordingly
    bo4_column = bo4_df.iloc[:, 1]  # Second column (index 1)
    ao3_column = ao3_df.iloc[:, 3]  # Fourth column (index 3)

    print("Bo4 Column Sample:")
    print(bo4_column.head())

    # Check if lengths match
    if len(bo4_column) != len(ao3_column):
        print(f"Warning: Columns have different lengths (BO4: {len(bo4_column)}, AO3: {len(ao3_column)})")
        return (False, None)
    
    # Compare the columns
    comparison = bo4_column == ao3_column
    
    if comparison.all():
        print("The columns are identical")
        return (True, None)
    else:
        print("The columns have differences")
        # Create a DataFrame showing differences
        diff_df = pd.DataFrame({
            'BO4_Value': bo4_column,
            'AO3_Value': ao3_column,
            'Match': comparison
        })
        # Filter to show only rows with differences
        differences = diff_df[~comparison].reset_index(drop=True)
        return (False, differences)

# Example usage
file_path = 'ARA_Region_Lenzburg.xls'
bo4_df = pd.read_excel(file_path, sheet_name=10)
ao3_df = pd.read_excel(file_path, sheet_name=3)
result = False#compare_excel_columns('ARA_Region_Lenzburg.xls')

if result is not None:
    are_identical, differences = result
    if not are_identical:
        print("\nDifferences found:")
        print(differences)