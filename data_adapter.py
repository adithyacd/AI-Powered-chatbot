import pandas as pd
import os

class DataAdapter:
    """
    Handles loading data from different file formats (CSV, Excel, etc.)
    """
    
    def load_data(self, file_path):
        """
        Load data from the given file path based on its extension
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_extension = file_path.split('.')[-1].lower()
        
        # Load data based on file type
        if file_extension == 'csv':
            return pd.read_csv(file_path)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
        elif file_extension == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def get_data_summary(self, dataframe):
        """
        Get a basic summary of the dataframe
        
        Args:
            dataframe (pandas.DataFrame): The data to summarize
            
        Returns:
            dict: Summary information about the dataframe
        """
        return {
            "shape": dataframe.shape,
            "columns": list(dataframe.columns),
            "data_types": {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            "has_missing_values": dataframe.isna().any().any()
        }