from enum import Enum

sonarqube_server = "http://localhost:9000"
sonarqube_token = "Your SonarQube login token"
sonarqube_username = "Your SonarQube login username"
sonarqube_password = "Your SonarQube login password"


class LLM_NAME(Enum):
    ChatGPT = "chatgpt"
    Copilot = "copilot"

    def __str__(self):
        return self.value


class LANGUAGE(Enum):
    Python = "python"
    Python_name = "Python"
    Java = "java"
    Java_name = "Java"
    CPP = "cpp"
    CPP_name = "C/C++"
    JavaScript = "javascript"
    JavaScript_name = "JavaScript"
    TypeScript = "typescript"
    TypeScript_name = "TypeScript"
    C = "c"
    CSharp = "csharp"
    CSharp_name = "C#"

    def __str__(self):
        return self.value


class KEYWORD(Enum):
    Generated = "generated"
    Authored = "authored"
    Coded = "coded"
    Created = "created"
    Implemented = "implemented"
    Written = "written"

    def __str__(self):
        return self.value


def transform_language_str(language_name):
    if language_name == LANGUAGE.Python.value:
        return LANGUAGE.Python_name.value
    elif language_name == LANGUAGE.Java.value:
        return LANGUAGE.Java_name.value
    elif language_name == LANGUAGE.CPP.value:
        return LANGUAGE.CPP_name.value
    elif language_name == LANGUAGE.JavaScript.value:
        return LANGUAGE.JavaScript_name.value
    elif language_name == LANGUAGE.TypeScript.value:
        return LANGUAGE.TypeScript_name.value

    return language_name


all_columns = ["index", "project_name", "create_time", "project_language", "contributor", "star", "fork", "issues",
               "watch", "project_commits", "code_language", "keyword_index", "matched_keyword",
               "url_of_first_file_commit",
               "path_of_first_file_commit", "final_change_commit_path", "code_granularity_data",
               "code_func_type", "number_of_bug_or_vulnerability_all_commit", "comments",
               "start_end_line_data", "first_loc", "final_start_end_line_data", "final_loc_add", "final_loc_minus",
               "final_change_type", "test_code_data", "regular_expression_data", "number_of_commits",
               "number_of_change_commit_to_first",
               "change_commit_to_first_index", "change_commit_to_first_hash", "change_commit_to_first_blocks",
               "number_of_all_change_commit",
               "all_change_commit_index", "all_change_commit_hash", "all_change_commit_blocks",
               "number_of_all_change_fix_commit", "real_fixed_commit_number", "real_fixed_commit_hash",
               "real_fixed_commit_reason", "all_change_fix_commit_index",
               "all_change_fix_commit_hash", "all_change_fix_commit_blocks", 'project_lines', 'project_locs',
               'project_statements', 'project_functions',
               'project_classes', 'project_files', 'project_density_comments', 'project_comments',
               'project_duplicated_lines', 'project_duplicated_blocks', 'project_duplicated_files',
               'project_duplicated_lines_density', "project_vulnerability", 'project_bugs', 'project_code_smells',
               'project_sqale_index', 'project_sqale_debt_ratio', 'project_complexity_all',
               'project_cognitive_complexity_all',
               'project_complexity_mean_method', 'project_cognitive_complexity_mean_method', 'code_lines', 'code_locs',
               'code_statements', 'code_functions',
               'code_classes', 'code_files', 'code_density_comments', 'code_comments', 'code_duplicated_lines',
               'code_duplicated_blocks', 'code_duplicated_files',
               'code_duplicated_lines_density', 'code_code_smells', 'code_sqale_index', 'code_sqale_debt_ratio',
               'code_complexity_all', 'code_cognitive_complexity_all',
               'code_complexity_mean_method', 'code_cognitive_complexity_mean_method', "manual_vs_gpt_loc_ratio",
               "manual_vs_gpt_change_commit_ratio", "manual_vs_gpt_change_fix_commit_ratio"]

if __name__ == "__main__":
    print(LLM_NAME.ChatGPT)
