import pandas as pd
import numpy as np

class Dataset:
    """
    A class to load, clean, and summarize CSV datasets.

    Attributes:
    -----------
    file_path : str
        Path to the CSV file.
    data : pd.DataFrame
        The loaded dataset.

    Methods:
    --------
    load()
        Loads the CSV file into a pandas DataFrame.
    clean(strategy='mean', columns=None, custom=0, inplace=True)
        Handles missing values with different strategies.
    summarize()
        Returns descriptive statistics and data types.
    """
    
    def __init__(self, file_path) -> None:
        
        """
        Initialize the Dataset object with a file path.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file to be loaded.
        """
        
        self.file_path = file_path
        self.data = None
    
    def load(self):
        
        """
        Load CSV file into a pandas DataFrame and store in self.data.

        Returns:
        --------
        pd.DataFrame
            The loaded dataset.
        """
        
        self.data = pd.read_csv(self.file_path)
        print("✅ Dataset loaded successfully")
        
        return self.data
    
    def clean(self, strategy="mean", columns=None, custom=0, inplace=True):
        
        """
        Handle missing values in the dataset.

        Parameters:
        -----------
        strategy : str, default='mean'
            Method to handle missing values. Options:
            - 'mean'   : fill with column mean
            - 'median' : fill with column median
            - 'custom' : fill with custom value provided in 'custom'
            - 'drop'   : drop rows with missing values
        columns : list, default=None
            List of columns to apply cleaning on. If None, applies to all columns.
        custom : numeric or str, default=0
            Custom value to fill when strategy='custom'.
        inplace : bool, default=True
            If True, modifies self.data; otherwise returns a new DataFrame.

        Returns:
        --------
        pd.DataFrame or None
            Returns cleaned DataFrame if inplace=False; otherwise modifies self.data in place.
        """

        if columns is None:
            columns = self.data.columns

        # Decide which DataFrame to work on
        df = self.data if inplace else self.data.copy()

        if strategy == "mean":
            for col in columns:
                df[col].fillna(df[col].mean(), inplace=True)

        elif strategy == "median":
            for col in columns:
                df[col].fillna(df[col].median(), inplace=True)

        elif strategy == "custom":
            for col in columns:
                df[col].fillna(custom, inplace=True)

        elif strategy == "drop":
            df.dropna(subset=columns, inplace=True)

        # Return the cleaned DataFrame if not inplace
        if not inplace:
            return df

    def summarize(self):
        """
        Generate descriptive statistics and column metadata for the dataset.

        Returns:
        --------
        tuple of pd.DataFrame:
            - statistics: Descriptive statistics for numerical columns
                (count, mean, std, min, 25%, 50%, 75%, max)
            - types: DataFrame containing:
                - Type: Data type of each column
                - Non-Null Count: Number of non-missing values per column

        Example:
        --------
        stats, col_info = financials_dataset.summarize()
        print(stats.head())
        print(col_info)
        """
        statistics = self.data.describe()  # already a DataFrame
        types = pd.DataFrame({
            "Type": self.data.dtypes,
            "Non-Null Count": self.data.notnull().sum()
        })
        
        return statistics, types


    def detect_numeric_financial(self, threshold=0.7):
        """
        Automatically detect numeric and financial columns in the dataset.

        This method scans each column to determine whether it contains numeric-like
        values or financial data (e.g., amounts with $ or parentheses). It updates
        the `numeric_columns` and `financial_columns` attributes for later processing.

        Parameters
        ----------
        threshold : float, default=0.7
            Minimum proportion of numeric-like values required to consider a column numeric.
            Columns below this threshold are treated as text.

        Updates
        -------
        self.numeric_columns : list
            List of columns auto-detected as numeric.
        self.financial_columns : list
            List of columns auto-detected as financial (contain symbols like $ or parentheses).

        Notes
        -----
        - Columns already stored as numeric types (int64, float64) are automatically included.
        - Columns with mixed text and numeric values below the threshold are ignored.
        """
        numeric_cols = []
        financial_cols = []

        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                sample = self.data[col].dropna().astype(str)

                # Detect financial symbols
                if sample.str.contains(r'[\$,()]').any():
                    financial_cols.append(col)
                    continue

                # Detect numeric-like strings
                numeric_like = sample.str.replace(r'[^0-9.]', '', regex=True)
                proportion_numeric = numeric_like.apply(lambda x: x.replace('.', '', 1).isnumeric()).mean()

                if proportion_numeric >= threshold:
                    numeric_cols.append(col)

        # Add existing numeric columns
        for col in self.data.select_dtypes(include=['int64', 'float64']).columns:
            if col not in numeric_cols:
                numeric_cols.append(col)

        self.numeric_columns = numeric_cols
        self.financial_columns = financial_cols

        print("Detected numeric columns:", self.numeric_columns)
        print("Detected financial columns:", self.financial_columns)
        

    def clean_currency(self, symbols=['$', ',', '/', ' '], inplace=True):
        """
        Clean financial/monetary columns by removing symbols and converting to numeric.

        This method works on columns detected as financial. It performs the following:
        1. Strips whitespace and replaces empty strings with NaN.
        2. Removes unwanted symbols like $, commas, /, and spaces.
        3. Converts parentheses into negative numbers, e.g., "(1000)" -> -1000.
        4. Converts cleaned strings to numeric (float).

        Parameters
        ----------
        symbols : list of str, default ['$' , ',' , '/' , ' ']
            Characters to remove from financial columns before numeric conversion.
        inplace : bool, default=True
            If True, modifies self.data directly; otherwise returns a cleaned DataFrame.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame (only if inplace=False).

        Notes
        -----
        - Automatically calls `detect_numeric_financial()` if financial columns are not identified.
        - Non-convertible values are replaced with NaN.
        """
        if not self.financial_columns:
            self.detect_numeric_financial()

        df = self.data if inplace else self.data.copy()

        # Clean column names
        df.columns = df.columns.str.strip()

        for col in self.financial_columns:
            df[col] = df[col].astype(str).str.strip().replace('', np.nan)

            # Remove unwanted symbols
            for symbol in symbols:
                df[col] = df[col].str.replace(symbol, '', regex=False)

            # Convert parentheses to negative numbers
            df[col] = df[col].str.replace(r'^\((.*)\)$', r'-\1', regex=True)

            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if inplace:
            self.data = df
        else:
            return df

        return df




    def convert_numeric_safe(self, columns=None, exclude_columns=None, inplace=True):
        
        """
        Safely convert numeric-like string columns to numeric types (int or float).

        Parameters
        ----------
        columns : list of str, optional
            Specific columns to convert. If None, all numeric-like columns detected
            by `detect_numeric_financial()` are considered.
        exclude_columns : list of str, optional
            Columns to exclude from conversion even if they contain numeric-like values.
        inplace : bool, default=True
            If True, modifies self.data directly; otherwise returns a new DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with converted numeric columns (only if inplace=False).

        Notes
        -----
        - Automatically calls `detect_numeric_financial()` if numeric columns are not identified.
        - Non-convertible values are replaced with NaN.
        - Pure text columns remain unchanged.
        """
        if exclude_columns is None:
            exclude_columns = []

        if not self.numeric_columns:
            self.detect_numeric_financial()

        if columns is None:
            columns = [col for col in self.numeric_columns if col not in exclude_columns]

        df = self.data if inplace else self.data.copy()

        for col in columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if inplace:
            self.data = df
        else:
            return df

        return df


    
    
        

        
        
        
        