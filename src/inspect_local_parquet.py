import pyarrow.parquet as pq
import sys

# Path to a local parquet file
parquet_path = "output/data/chunk-000/file-000.parquet"

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
    
    # Print first few rows of data
    print("\nFirst 5 rows of data:")
    # Convert to pandas only for display purposes, without assigning to a variable
    print(table.slice(0, 5).to_pandas())
    
    # Print basic information about each column
    # print("\nColumn information:")
    # for i, name in enumerate(table.column_names):
    #     column = table.column(i)
    #     print(f"  {name}:")
    #     print(f"    Type: {column.type}")
    #     print(f"    Null count: {column.null_count}")
        
except Exception as e:
    print(f"Error reading parquet file: {e}")
    import traceback
    traceback.print_exc()
