import argparse
from analyze import write_code_analysis_result_to_csv


def main():
    parser = argparse.ArgumentParser(description='Command-line interface for running tasks.')

    subparsers = parser.add_subparsers(dest='command', help='sub-command help')

    # command: analysis
    analyze_parser = subparsers.add_parser('analyze', help='analyze github code generated by LLms')
    analyze_parser.add_argument('--llm', type=str, required=True, help='llm name', choices=["chatgpt", "copilot"])
    analyze_parser.add_argument('--lang', type=str, required=True, help='language', choices=["python", "java"])
    analyze_parser.add_argument('--keyword', type=str, required=True, help='keyword', choices=["generated by chatgpt"])

    args = parser.parse_args()

    if args.command == 'analyze':
        write_code_analysis_result_to_csv(args.llm, args.lang, args.keyword)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
