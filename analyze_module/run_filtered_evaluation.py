from filtered_evaluation.evaluator import FilteredHumanSystemEvaluator

def main():
    system_output_dir = "path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/alarm_result_output/alarm_result_output_v13_anm"
    human_evaluation_filtered_dir = "path_to_your_analysis_root/SNH/0_HUMAN_RESULTS/Combo_Evaluators/human_evaluation_combo_filtered/human_evaluation_unified_e1_e3_e5_e4_e2"
    output_dir = "path_to_your_analysis_root/SNH/0_Evaluation_Result/human_system_evaluation_results"
    data_splits_path = "path_to_your_analysis_root/SNH/data_splits.json"
    valid_time_interval_path = "video_time_periods.csv"

    included_ids = set()
    excluded_ids = set()
    date_filter = None
    
    evaluator = FilteredHumanSystemEvaluator( 
        system_output_dir=system_output_dir,
        human_evaluation_filtered_dir=human_evaluation_filtered_dir,
        output_dir=output_dir,
        data_splits_path=data_splits_path,
        valid_time_interval_path=valid_time_interval_path,
        included_ids=included_ids,
        excluded_ids=excluded_ids,
        date_filter=date_filter
    )
    
    evaluator.run_full_evaluation()

if __name__ == '__main__':
    main() 