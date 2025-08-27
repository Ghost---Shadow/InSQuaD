import os
import glob
import pandas as pd
from pathlib import Path


def merge_csv_to_excel(pattern="./tables_*.csv", output_file="merged_tables.xlsx"):
    """
    Merge multiple CSV files into a single Excel file.
    Each CSV becomes a separate sheet named after the CSV filename.

    Args:
        pattern (str): Glob pattern to match CSV files
        output_file (str): Name of the output Excel file
    """
    # Define the specific column order for selected columns
    column_order = [
        "mrpc",
        "sst5",
        "mnli",
        "dbpedia",
        "rte",
        "hellaswag",
        "mwoz",
        "geoq",
        "xsum",
    ]

    # Get list of all CSV files matching the pattern
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for csv_file in csv_files:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Reorder columns while preserving other columns
            # First, get all columns that are not in our specified order
            all_columns = df.columns.tolist()

            # Create a new column order list
            new_column_order = []

            # Add all columns that are not in our specified order first
            for col in all_columns:
                if col not in column_order:
                    new_column_order.append(col)

            # Add specified columns in the given order (if they exist in the dataframe)
            for col in column_order:
                if col in all_columns:
                    new_column_order.append(col)

            # Reorder the dataframe columns
            df = df[new_column_order]

            # Get the filename without path and extension to use as sheet name
            sheet_name = Path(csv_file).stem

            # Remove "tables_" prefix if present for cleaner sheet names
            sheet_name = sheet_name.replace("tables_", "")

            # Excel has a 31 character limit for sheet names
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]

            # Write the dataframe to a sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Added {csv_file} as sheet '{sheet_name}' with reordered columns")

    print(f"Successfully created Excel file: {output_file}")


if __name__ == "__main__":
    merge_csv_to_excel()
