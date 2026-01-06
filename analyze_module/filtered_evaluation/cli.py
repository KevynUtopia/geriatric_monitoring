import argparse
from .evaluator import FilteredHumanSystemEvaluator

def main():
    parser = argparse.ArgumentParser(description='Filtered Human vs System Evaluation')
    parser.add_argument('--system_output_dir', type=str, required=True, help='Directory containing system output files')
    parser.add_argument('--human_evaluation_filtered_dir', type=str, required=True, help='Directory containing filtered human evaluation subfolders')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--data_splits_path', type=str, required=True, help='Path to data_splits.json file')
    args = parser.parse_args()

    evaluator = FilteredHumanSystemEvaluator(
        system_output_dir=args.system_output_dir,
        human_evaluation_filtered_dir=args.human_evaluation_filtered_dir,
        output_dir=args.output_dir,
        data_splits_path=args.data_splits_path
    )
    evaluator.run_full_evaluation()

if __name__ == '__main__':
    main() 