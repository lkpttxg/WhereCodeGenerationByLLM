import pandas as pd

from config import config, all_columns
from utils import get_files_and_path


def read_xlsx_to_csv(xlsx_path, llm_name, language, is_change=False):
    filenames, filepaths = get_files_and_path(config[llm_name][language]['res'])
    filename = filenames[0]
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
        # 找到 df 中对应 path_of_first_file_commit 的行
        df_indexes = df[df['path_of_first_file_commit'] == path_of_first_file_commit].index
        if not df_indexes.empty:
            if is_change:
                code_change_type = row['code_change_type']
                for df_index in df_indexes:
                    df.at[df_index, 'final_change_type'] = code_change_type
            else:
                # 读取 xlsx_df 行的 code_type 内容并填充到 df
                code_type_value = row['code_type']
                # code_type_value = row['code_type_new_type']
                for df_index in df_indexes:
                    # df.at[df_index, 'code_func_type'] = code_type_value
                    if df.loc[df_index, 'regular_expression_data'] in ['input', 'regular expression']:
                        df.at[df_index, 'code_func_type'] = "Program input code"
                        # df.at[df_index, 'code_func_type'] = "Program input code, i.e., variable assignments and regular expressions"
                    elif df.loc[df_index, 'test_code_data'] in ['test code']:
                        df.at[df_index, 'code_func_type'] = "Testing and debugging"
                        # df.at[df_index, 'code_func_type'] = "Testing and debugging"
                    else:
                        df.at[df_index, 'code_func_type'] = code_type_value

    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print("===========Save File: csv================")
    print("Analysis results have been saved to:", filepath)