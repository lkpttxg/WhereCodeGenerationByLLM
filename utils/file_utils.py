import pandas as pd
from config import config, all_columns
from utils import get_files_and_path


def read_xlsx_to_csv(xlsx_path, llm_name, language, is_change=False):
    """
        Reads data from an Excel file and updates a CSV file based on the content of the Excel file.

        Args:
            xlsx_path (str): The file path to the Excel file to read.
            llm_name (str): The name of the language model.
            language (str): The programming language.
            is_change (bool): Flag indicating whether to update the 'final_change_type' or 'code_func_type'.

        Returns:
            None
    """
    _, filepaths = get_files_and_path(config[llm_name][language]['res'])
    filepath = filepaths[0]
    print("Reading xlsx file:", xlsx_path)
    xlsx_df = pd.read_excel(xlsx_path)
    print("Reading csv file:", filepath)

    old_df = pd.read_csv(filepath, encoding='utf-8')
    df = pd.DataFrame(columns=all_columns)
    for col in old_df.columns:
        if col in all_columns:
            df[col] = old_df[col]

    for index, row in xlsx_df.iterrows():
        path_of_first_file_commit = row['path_of_first_file_commit']
        # Find the line in df that corresponds to path_of_first_file_commit
        df_indexes = df[df['path_of_first_file_commit'] == path_of_first_file_commit].index
        if not df_indexes.empty:
            if is_change:
                code_change_type = row['code_change_type']
                for df_index in df_indexes:
                    df.at[df_index, 'final_change_type'] = code_change_type
            else:
                # Read the code_type content of the xlsx_df line and fill it into df
                code_type_value = row['code_type']
                for df_index in df_indexes:
                    if df.loc[df_index, 'regular_expression_data'] in ['input', 'regular expression']:
                        df.at[df_index, 'code_func_type'] = "Program input code"
                    elif df.loc[df_index, 'test_code_data'] in ['test code']:
                        df.at[df_index, 'code_func_type'] = "Testing and debugging"
                    else:
                        df.at[df_index, 'code_func_type'] = code_type_value

    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print("===========Save File: csv================")
    print("Analysis results have been saved to:", filepath)