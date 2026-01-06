from filtered_evaluation.evaluator import FilteredHumanSystemEvaluator

def main():
    # Hardcoded config
    system_output_dir = "path_to_your_analysis_root/SNH/0_SYSTEM_RESULTS/alarm_result_output/alarm_result_output_v13_anm"
    # human_evaluation_filtered_dir = "path_to_your_analysis_root/SNH/human_evaluation_filtered"
    '''
        Available evaluator combinations (anonymized: e1=cyy, e2=zzr, e3=hsy, e4=zww, e5=roy):
        human_evaluation_unified_e1_e2
        human_evaluation_unified_e3_e2
        human_evaluation_unified_e5_e2
        human_evaluation_unified_e4_e2
        human_evaluation_unified_e1_e3_e5_e4_e2
    '''
    
    human_evaluation_filtered_dir = "path_to_your_analysis_root/SNH/0_HUMAN_RESULTS/Combo_Evaluators/human_evaluation_combo_filtered/human_evaluation_unified_e1_e3_e5_e4_e2"
    output_dir = "path_to_your_analysis_root/SNH/0_Evaluation_Result/human_system_evaluation_results"
    data_splits_path = "path_to_your_analysis_root/SNH/data_splits.json"  # Update this path to match your local structure
    valid_time_interval_path = "video_time_periods.csv"

    # --- Identity filtering examples ---
    # Option 1: Set filters during initialization
    included_ids = set()  # Empty set = evaluate all identities
    # included_ids = {'p_3', 'p_8', 'p_11', 'p_16', 'p_19', 'p_20', 'p_23', 'p_24'}  # Only evaluate specific identities
    
    excluded_ids = set()  # Empty set = no exclusions
    # excluded_ids = {'p_13', 'p_15', 'p_22'}  # Exclude specific identities
    
    # --- Date filtering examples ---
    # Option 1: Set date filter during initialization
    # date_filter = "06_26"  # None = evaluate all dates
    # date_filter = "06_26"  # Only evaluate files from June 26th, 2019
    # date_filter = "07_15"  # Only evaluate files from July 15th, 2019
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
    
    # Option 2: Set filters after initialization
    # evaluator.set_identity_filters(
    #     included_ids={'p_1', 'p_2', 'p_3'},  # Only these identities
    #     excluded_ids={'p_13', 'p_15'}         # Exclude these identities
    # )
    
    # Option 3: Set date filter after initialization
    # evaluator.set_date_filter("06_26")  # Only evaluate files from June 26th, 2019
    # evaluator.set_date_filter("07_15")  # Only evaluate files from July 15th, 2019
    # evaluator.set_date_filter(None)     # Evaluate all dates
    
    # Option 4: Print available identities for reference
    # evaluator.print_available_identities()
    
    evaluator.run_full_evaluation()

if __name__ == '__main__':
    main() 