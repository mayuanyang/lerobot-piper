import pyarrow.parquet as pq
import sys

# Path to a local parquet file
parquet_path = "output/data/chunk-001/file-001.parquet"

try:
    # Read the parquet file
    table = pq.read_table(parquet_path)
    
    # Print schema information
    print("Schema of the local parquet file:")
    print(table.schema)
    
    # Print some basic information about the table
    print("\nTable information:")
    print(f"Number of rows: {table.num_rows}")
    print(f"Number of columns: {table.num_columns}")
    
    # Print column names
    print("\nColumn names:")
    for i, name in enumerate(table.column_names):
        print(f"  {i}: {name}")
        
except Exception as e:
    print(f"Error reading parquet file: {e}")
    import traceback
    traceback.print_exc()
