from integrate_system.system_enging import SystemEngine
from integrate_system.arg_parser import parse_args
import os
import matplotlib.pyplot as plt
import pandas as pd

def main(args):
    # Initialize the system engine
    engine = SystemEngine(args.csv_path)
    
    # Smooth the data
    engine.smooth_data(sigma=args.sigma)
    

    if args.gantt_chart:
        # Create and get the gantt chart
        fig = engine.get_gantt_chart(palette_name=args.color_palette)
        # save the figure+

        fig.savefig(os.path.join(args.save_path, 'gantt_chart.jpg'), format='jpg')

    alert_curve, composite_score = engine.factor_analysis()
    # alert_curve is an array of 0 and 1 matches with df
    # now, extract the values of df['timestamp'] when alert_curve is 1
    alert_timestamps = engine.df[alert_curve == 1].index
    print(alert_timestamps)
    # save the alert_timestamps to a csv file
    pd.Series(alert_timestamps).to_csv(os.path.join(args.save_path, 'alert_timestamps.csv'), index=False)


    

if __name__ == "__main__":
    args = parse_args()
    main(args)





