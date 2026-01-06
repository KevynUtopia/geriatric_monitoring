import argparse

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - csv_path: Path to the CSV file
            - color_palette: Color palette to use for visualization
            - verbose: Flag for verbose output
    """
    parser = argparse.ArgumentParser(description='System Engine for Data Processing')
    
    # Add arguments
    parser.add_argument('--csv_path', 
                       type=str,
                       default='/Users/kevynzhang/Downloads/soft_v2/PID/p_1/06_25_8_00_am.csv',
                       help='Path to the CSV file')
    
    parser.add_argument('--color_palette',
                       type=str,
                       default='default',
                       choices=['default', 'pastel', 'vibrant', 'nature', 'sunset', 
                               'ocean', 'autumn', 'winter', 'spring', 'monochrome'],
                       help='Color palette to use for visualization')
    
    parser.add_argument('--verbose',
                       action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--sigma',
                       type=float,
                       default=3.0,
                       help='Standard deviation for Gaussian filter smoothing')
    
    parser.add_argument('--cmaps',
                       type=str,
                       default='default',
                       help='Color palette to use for visualization')
    
    parser.add_argument('--save_path',
                       type=str,
                       default='/Users/kevynzhang/Downloads/system_output',
                       help='Path to save the results')
    
    parser.add_argument('--gantt_chart',
                       action='store_true',
                       default=False,
                       help='Generate gantt chart')
    
    return parser.parse_args() 