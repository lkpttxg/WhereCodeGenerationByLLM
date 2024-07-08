from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import config, LANGUAGE, LLM_NAME, transform_language_str
from utils import get_files_and_path, parse_str_to_arr

columns=('contributor', 'star', 'fork', 'issues', 'watch',
                                                 'project_commits', 'number_of_bug_or_vulnerability_all_commit',
                                                 'first_loc', 'number_of_commits', 'project_files', 'project_lines',
                                                 'project_locs', 'project_statements', 'project_functions',
                                                 'project_code_smells',
                                                 'project_complexity_all', 'project_cognitive_complexity_all',
                                                 'code_lines', 'code_locs', 'code_statements', 'code_functions',
                                                 'code_code_smells', 'code_complexity_all',
                                                 'code_cognitive_complexity_all')
titles=['Contributors Distribution',
                        'Stars Distribution',
                        'Forks Distribution',
                        'Issues Distribution',
                        'Watchers Distribution',
                        'Project Commits Distribution',
                        'Bug/Vulnerability Commits Distribution',
                        'Generated Code LOC Distribution',
                        'File Commits Distribution',
                        'Project Files Distribution',
                        'Project Lines Distribution',
                        'Project LOCs Distribution',
                        'Project Statements Distribution',
                        'Project Functions Distribution',
                        'Project Code Smells Distribution',
                        'Project Complexity Distribution',
                        'Project Cognitive Complexity Distribution',
                        'Code Lines Distribution',
                        'Code LOCs Distribution',
                        'Code Statements Distribution',
                        'Code Functions Distribution',
                        'Code Code Smells Distribution',
                        'Code Complexity Distribution',
                        'Code Cognitive Complexity Distribution'
                    ]
y_labels=[
            'Number of Contributors',
            'Number of Stars',
            'Number of Forks',
            'Number of Issues',
            'Number of Watchers',
            'Number of Project Commits',
            'Number of Bug/Vulnerability Commits',
            'Generated Code LOC',
            'Number of File Commits',
            'Number of Project Files',
            'Number of Project Lines',
            'Number of Project LOCs',
            'Number of Project Statements',
            'Number of Project Functions',
            'Number of Project Code Smells',
            'Project Complexity',
            'Project Cognitive Complexity',
            'Number of Code Lines',
            'Number of Code LOCs',
            'Number of Code Statements',
            'Number of Code Functions',
            'Number of Code Code Smells',
            'Code Complexity',
            'Code Cognitive Complexity'
        ]



def draw_violinplot_from_different_metrics(cols, titles=None, y_labels=None, res_name="", is_project=False, top_percent=None):
    if not isinstance(cols, list):
        if isinstance(cols, str):
            cols = [cols]
        else:
            raise ValueError("cols must be a list of strings or a single string")

    Language_final_csv_path = {
        LANGUAGE.Python.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.Python.value]['res'])[1][0],
        LANGUAGE.Java.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.Java.value]['res'])[1][0],
        LANGUAGE.CPP.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.CPP.value]['res'])[1][0],
        LANGUAGE.JavaScript.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.JavaScript.value]['res'])[1][0],
        LANGUAGE.TypeScript.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.TypeScript.value]['res'])[1][0],
    }


    # Set drawing style
    sns.set(style="whitegrid")
    # Initialize the subgraph
    num_columns = len(Language_final_csv_path)
    num_rows = len(cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(22, 3 * num_rows))
    axs = axs if isinstance(axs, np.ndarray) and axs.ndim > 1 else np.expand_dims(axs, axis=0)

    for row_idx, col in enumerate(cols):
        for col_idx, (language, filepath) in enumerate(Language_final_csv_path.items()):
            print("Reading csv file:", filepath)
            df = pd.read_csv(filepath)
            if is_project:
                df_dop = df.drop_duplicates(subset=['project_name'])
            else:
                df_dop = df

            clean_data = df_dop[col].dropna()
            filtered_data = clean_data

            # --- filter 1 -----
            # Calculate the 99.5% quantile
            # threshold = np.percentile(clean_data, 99.5)
            # # Filter out points that are less than or equal to the quantile threshold
            # filtered_data = clean_data[clean_data <= threshold]

            # --- filter 2 -----
            # Remove outliers using IQR method
            # Q1 = clean_data.quantile(0.25)
            # Q3 = clean_data.quantile(0.75)
            # IQR = Q3 - Q1
            # lower_bound = Q1 - 100 * IQR
            # upper_bound = Q3 + 100 * IQR
            # filtered_data = clean_data[(clean_data >= lower_bound) & (clean_data <= upper_bound)]

            # --- filter 3 -----
            # Calculate the mean and standard deviation
            # mean = clean_data.mean()
            # std = clean_data.std()
            # # Select points that are less than or equal to twice the mean standard deviation
            # threshold = mean + 2 * std
            # filtered_data = clean_data[clean_data <= threshold]

            # --- filter 4 -----
            # Threshold setting
            # threshold = 50
            # # Filter out the points that are less than or equal to the threshold
            # filtered_data = clean_data[clean_data <= threshold]

            # --- filter 5 -----
            # # Sort
            # sorted_data = clean_data.sort_values(ascending=False)
            # # Retrieves the index of the first N highest values
            # N = 3  # Remove the highest 3 values
            # indices_to_remove = sorted_data.index[:N]
            # # Remove the highest N values
            # filtered_data = clean_data.drop(indices_to_remove)

            sns.violinplot(y=filtered_data, ax=axs[row_idx, col_idx], cut=0)
            axs[row_idx, col_idx].set_ylim(bottom=-1)
            # Set the font size of the Y-axis scale label
            axs[row_idx, col_idx].tick_params(axis='y', labelsize=20)
            # Calculate statistics
            median = filtered_data.median()
            mean = filtered_data.mean()
            min_val = filtered_data.min()
            max_val = filtered_data.max()
            std_dev = filtered_data.std()

            # Add statistics to the plot
            axs[row_idx, col_idx].axhline(median, color='k', linestyle='--', label='Median', linewidth=2)
            axs[row_idx, col_idx].axhline(mean, color='r', linestyle='-', label='Mean', linewidth=2)
            axs[row_idx, col_idx].axhline(min_val, color='y', linestyle=':', label='Min', linewidth=2)
            axs[row_idx, col_idx].axhline(max_val, color='g', linestyle=':', label='Max', linewidth=2)
            axs[row_idx, col_idx].fill_betweenx([mean - std_dev, mean + std_dev], -0.5, 0.5, color='teal', alpha=0.2, label='Std Dev')

            # Add text annotations for statistics
            ylims = axs[row_idx, col_idx].get_ylim()
            y_range = ylims[1] - ylims[0]
            vertical_spacing = y_range * 0.1

            font_size = 18

            axs[row_idx, col_idx].text(0.50, mean + vertical_spacing, f'Mean: {mean:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='r', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.50, median + vertical_spacing*0.1, f'Median: {median:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='k', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.95, min_val + vertical_spacing*0.1, f'Min: {min_val:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='y', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.95, max_val - vertical_spacing, f'Max: {max_val:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='g', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.95, mean + std_dev + vertical_spacing*1.5, f'Std Dev: {std_dev:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='teal', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())

            # Set custom title and y-label if provided
            # if titles and col_idx == 2:
            #     axs[row_idx, col_idx].set_title(titles[row_idx], fontsize=16)


            if y_labels and col_idx == 0:
                axs[row_idx, col_idx].set_ylabel(y_labels[row_idx], fontsize=20)
            else:
                axs[row_idx, col_idx].set_ylabel(None)

            if row_idx == num_rows - 1:
                axs[row_idx, col_idx].set_xlabel(transform_language_str(language), fontsize=20)

            if row_idx == 0 and col_idx == 0:
                axs[row_idx, col_idx].legend(prop={'size': 'large'})

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    fig_save_path = Path(config[LLM_NAME.ChatGPT.value]['visualization']['res']) / f"project_metrics_{res_name}.pdf"
    plt.savefig(fig_save_path)
    print("Violinplot saved to:", fig_save_path)


def draw_violinplot_manual_vs_gpt(top_percent=None):
    Language_final_csv_path = {
        LANGUAGE.Python.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.Python.value]['res'])[1][0],
        LANGUAGE.Java.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.Java.value]['res'])[1][0],
        LANGUAGE.CPP.value: get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.CPP.value]['res'])[1][0],
        LANGUAGE.JavaScript.value:
            get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.JavaScript.value]['res'])[1][0],
        LANGUAGE.TypeScript.value:
            get_files_and_path(config[LLM_NAME.ChatGPT.value][LANGUAGE.TypeScript.value]['res'])[1][0],
    }

    cols = ["written / generated LOC", "written / generated method CC", "written / generated method CogC",
            "written / generated mod.", "written / generated bug mod."]

    y_labels = ["The proportion\nof generated code", "The ratio of average\nmethod CC", "The ratio of average\nmethod CogC", "The ratio of average\nmodifications", "The ratio of average\nbug-fix modifications"]

    # Set drawing style
    sns.set(style="whitegrid")
    # Initialize the subgraph
    num_columns = len(Language_final_csv_path)
    num_rows = len(cols)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(22, 3 * num_rows))
    axs = axs if isinstance(axs, np.ndarray) and axs.ndim > 1 else np.expand_dims(axs, axis=0)

    for row_idx, col in enumerate(cols):
        for col_idx, (language, filepath) in enumerate(Language_final_csv_path.items()):
            print("Reading csv file:", filepath)
            df = pd.read_csv(filepath)

            if top_percent:
                remove_dup_project_df = df.drop_duplicates(subset=['project_name'])
                top_percent_threshold = remove_dup_project_df['star'].quantile(1 - top_percent)
                print("top_percent_threshold", top_percent_threshold)
                df = df[df['star'] >= top_percent_threshold]

            if col == "written / generated LOC":
                clean_df = df.dropna(subset=["project_functions"])
                generated_loc = clean_df["code_locs"]
                manual_loc = clean_df["project_locs"]

                filtered_data = generated_loc*1.0 / manual_loc

                # Gets the index corresponding to the maximum value in filtered_data
                max_index = filtered_data.idxmax()

                # Gets the row information in clean_df corresponding to max_index
                max_row_info = clean_df.loc[max_index]

                print("filtered_data Specifies the row corresponding to the maximum value:")
                print(max_row_info)

                # filtered_data = pd.Series(
                #     np.where(
                #         (manual_loc-generated_loc) != 0,
                #         generated_loc * 1.0 / (manual_loc-generated_loc),
                #         1
                #     ),
                #     index=clean_df.index
                # )

            elif col == "written / generated method CC":
                clean_df = df.dropna(subset=["project_functions"])
                clean_df = clean_df[(clean_df["code_functions"] > 0) & (clean_df["project_functions"] > 0)]

                generated_cc = clean_df["code_complexity_all"]
                generated_fuc = clean_df["code_functions"]
                generated_ratio_method = generated_cc*1.0 / generated_fuc

                manual_cc = clean_df["project_complexity_all"]
                manual_fuc = clean_df["project_functions"]
                # manual_ratio_method = (manual_cc) * 1.0 / (manual_fuc)
                manual_ratio_method = pd.Series(
                    np.where(
                        (manual_fuc - generated_fuc) != 0,
                        (manual_cc - generated_cc) * 1.0 / (manual_fuc - generated_fuc),
                        np.nan
                    ),
                    index=clean_df.index
                )

                valid_indices = (manual_fuc - generated_fuc) != 0

                manual_ratio_method = manual_ratio_method[valid_indices]
                generated_ratio_method = generated_ratio_method[valid_indices]

                special_condition_indices = (manual_ratio_method == 0) & (generated_ratio_method != 0)
                special_condition_indices = special_condition_indices.reindex(clean_df.index, fill_value=False)
                filtered_special_condition_data = clean_df[special_condition_indices]

                if len(filtered_special_condition_data) > 0:
                    print(
                        f"============language:{language}, project mean CC = 0, but generated code != 0================")
                    print("length special condition data:", len(filtered_special_condition_data))
                    print(filtered_special_condition_data['index'])

                manual_ratio_method = manual_ratio_method[~special_condition_indices]
                generated_ratio_method = generated_ratio_method[~special_condition_indices]

                filtered_data = generated_ratio_method / manual_ratio_method

                # Gets the index corresponding to the maximum value in filtered_data
                max_index = filtered_data.idxmax()

                # Gets the row information in clean_df corresponding to max_index
                max_row_info = clean_df.loc[max_index]

                # Output line information
                print("filtered_data Specifies the row corresponding to the maximum value:")
                print(max_row_info)

            elif col == "written / generated method CogC":
                clean_df = df.dropna(subset=["project_functions"])
                clean_df = clean_df[(clean_df["code_functions"] > 0) & (clean_df["project_functions"] > 0)]

                generated_cogc = clean_df["code_cognitive_complexity_all"]
                generated_fuc = clean_df["code_functions"]
                generated_ratio_method = generated_cogc*1.0 / generated_fuc

                manual_cogc = clean_df["project_cognitive_complexity_all"]
                manual_fuc = clean_df["project_functions"]

                # manual_ratio_method = (manual_cogc) * 1.0 / (manual_fuc)

                manual_ratio_method = pd.Series(
                    np.where(
                        (manual_fuc - generated_fuc) != 0,
                        (manual_cogc - generated_cogc) * 1.0 / (manual_fuc - generated_fuc),
                        np.nan
                    ),
                    index=clean_df.index
                )

                valid_indices = (manual_fuc - generated_fuc) != 0

                manual_ratio_method = manual_ratio_method[valid_indices]
                generated_ratio_method = generated_ratio_method[valid_indices]

                special_condition_indices = (manual_ratio_method == 0) & (generated_ratio_method != 0)
                special_condition_indices = special_condition_indices.reindex(clean_df.index, fill_value=False)
                filtered_special_condition_data = clean_df[special_condition_indices]

                if len(filtered_special_condition_data) > 0:
                    print(f"============language:{language}, project mean CngC = 0, but generated code != 0================")
                    print("length special condition data:", len(filtered_special_condition_data))
                    print(filtered_special_condition_data['index'])
                    print(filtered_special_condition_data['path_of_first_file_commit'].tolist())

                manual_ratio_method = manual_ratio_method[~special_condition_indices]
                generated_ratio_method = generated_ratio_method[~special_condition_indices]

                filtered_data = generated_ratio_method / manual_ratio_method

                max_index = filtered_data.idxmax()

                max_row_info = clean_df.loc[max_index]

                print("filtered_data Specifies the row corresponding to the maximum value:")
                print(max_row_info)
            elif col == "written / generated mod.":
                clean_df = df

                generated_mod = df["number_of_all_change_commit"]
                generated_file = pd.Series(1, index=df.index)
                manual_mod = df["project_commits"]
                manual_file = df["project_files"]

                # special_condition_indices = (generated_mod == manual_mod)
                # special_condition_indices = special_condition_indices.reindex(clean_df.index, fill_value=False)

                # filtered_special_condition_data = clean_df[special_condition_indices]
                # if len(filtered_special_condition_data) > 0:
                #     print(f"============language:{language}, generated_mod == manual_mod================")
                #     print("length special condition data:", len(filtered_special_condition_data))
                #     print(filtered_special_condition_data['index'])
                #     print(filtered_special_condition_data['path_of_first_file_commit'].tolist())

                # generated_mod = generated_mod[~special_condition_indices]
                # generated_file = generated_file[~special_condition_indices]
                # manual_mod = manual_mod[~special_condition_indices]
                # manual_file = manual_file[~special_condition_indices]
                #
                # generated_mod = generated_mod.reindex(clean_df.index, fill_value=np.nan)
                # generated_file = generated_file.reindex(clean_df.index, fill_value=np.nan)
                # manual_mod = manual_mod.reindex(clean_df.index, fill_value=np.nan)
                # manual_file = manual_file.reindex(clean_df.index, fill_value=np.nan)

                filtered_data = pd.Series(
                    np.where(
                        (manual_file - generated_file).notna() & (manual_file - generated_file != 0),
                        generated_mod*1.0 / (manual_mod * 1.0 / (manual_file - generated_file)),
                        generated_mod / manual_mod
                    ),
                    index=clean_df.index
                )


                max_index = filtered_data.idxmax()

                max_row_info = clean_df.loc[max_index]

                print("filtered_data Specifies the row corresponding to the maximum value:")
                print(max_row_info)

            elif col == "written / generated bug mod.":
                clean_df = df

                generated_mod = df["real_fixed_commit_number"].replace(np.nan, 0)
                generated_file = pd.Series(1, index=df.index)
                manual_mod = df["number_of_bug_or_vulnerability_all_commit"]
                manual_file = df["project_files"]

                # special_condition_indices = (generated_mod == manual_mod)
                # special_condition_indices = special_condition_indices.reindex(clean_df.index, fill_value=False)
                #
                # filtered_special_condition_data = clean_df[special_condition_indices]
                # if len(filtered_special_condition_data) > 0:
                #     print(f"============language:{language}, generated_mod == manual_mod================")
                #     print("length special condition data:", len(filtered_special_condition_data))
                #     print(filtered_special_condition_data['index'])
                #     print(filtered_special_condition_data['path_of_first_file_commit'].tolist())
                #
                # generated_mod = generated_mod[~special_condition_indices]
                # generated_file = generated_file[~special_condition_indices]
                # manual_mod = manual_mod[~special_condition_indices]
                # manual_file = manual_file[~special_condition_indices]
                #
                # generated_mod = generated_mod.reindex(clean_df.index, fill_value=np.nan)
                # generated_file = generated_file.reindex(clean_df.index, fill_value=np.nan)
                # manual_mod = manual_mod.reindex(clean_df.index, fill_value=np.nan)
                # manual_file = manual_file.reindex(clean_df.index, fill_value=np.nan)

                filtered_data = pd.Series(
                    np.where(
                        (manual_file - generated_file).notna() & (manual_file - generated_file != 0),
                        generated_mod / ((manual_mod) * 1.0 / (manual_file - generated_file)),
                        generated_mod / manual_mod
                    ),
                    index=clean_df.index
                )


                max_index = filtered_data.idxmax()


                max_row_info = clean_df.loc[max_index]


                print("filtered_data Specifies the row corresponding to the maximum value:")
                print(max_row_info)


            sns.violinplot(y=filtered_data, ax=axs[row_idx, col_idx], cut=0)
            axs[row_idx, col_idx].set_ylim(bottom=0)
            axs[row_idx, col_idx].tick_params(axis='y', labelsize=20)
            # Calculate statistics
            median = filtered_data.median()
            mean = filtered_data.mean()
            min_val = filtered_data.min()
            max_val = filtered_data.max()
            std_dev = filtered_data.std()

            # Add statistics to the plot
            axs[row_idx, col_idx].axhline(median, color='k', linestyle='--', label='Median', linewidth=2)
            axs[row_idx, col_idx].axhline(mean, color='r', linestyle='-', label='Mean', linewidth=2)
            axs[row_idx, col_idx].axhline(min_val, color='y', linestyle=':', label='Min', linewidth=2)
            axs[row_idx, col_idx].axhline(max_val, color='g', linestyle=':', label='Max', linewidth=2)
            axs[row_idx, col_idx].fill_betweenx([mean - std_dev, mean + std_dev], -0.5, 0.5, color='teal', alpha=0.2,
                                                label='Std Dev')

            # Add text annotations for statistics
            ylims = axs[row_idx, col_idx].get_ylim()
            y_range = ylims[1] - ylims[0]
            vertical_spacing = y_range * 0.1

            font_size = 18

            axs[row_idx, col_idx].text(0.50, mean + vertical_spacing, f'Mean: {mean:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='r', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.50, median + vertical_spacing * 0.1, f'Median: {median:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='k', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.95, min_val + vertical_spacing * 0.3, f'Min: {min_val:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='y', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.95, max_val - vertical_spacing, f'Max: {max_val:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='g', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())
            axs[row_idx, col_idx].text(0.95, mean + std_dev + vertical_spacing * 1.5, f'Std Dev: {std_dev:.2f}',
                                       horizontalalignment='right',
                                       verticalalignment='center', size=font_size, color='teal', weight='semibold',
                                       transform=axs[row_idx, col_idx].get_yaxis_transform())

            # Set custom title and y-label if provided
            # if titles and col_idx == 2:
            #     axs[row_idx, col_idx].set_title(titles[row_idx], fontsize=16)

            if y_labels and col_idx == 0:
                axs[row_idx, col_idx].set_ylabel(y_labels[row_idx], fontsize=20)
            else:
                axs[row_idx, col_idx].set_ylabel(None)

            if row_idx == num_rows - 1:
                axs[row_idx, col_idx].set_xlabel(transform_language_str(language), fontsize=20)

            if row_idx == 0 and col_idx == 0:
                axs[row_idx, col_idx].legend(prop={'size': 'large'})

    plt.subplots_adjust(hspace=0.5, wspace=1.5)
    # plt.tight_layout(pad=1.5, h_pad=0, w_pad=0)
    plt.tight_layout(h_pad=0, w_pad=2)
    fig_save_path = Path(config[LLM_NAME.ChatGPT.value]['visualization']['res']) / f"manual_vs_generated_RQ3_percent_{top_percent}.pdf"
    plt.savefig(fig_save_path)
    print("Violinplot saved to:", fig_save_path)
