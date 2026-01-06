from integrate_system.system_enging import SystemEngine
from integrate_system.arg_parser import parse_args
import os
import pandas as pd

def main(args):
    engine = SystemEngine(args.csv_path)
    engine.smooth_data(sigma=args.sigma)

    if args.gantt_chart:
        fig = engine.get_gantt_chart(palette_name=args.color_palette)
        fig.savefig(os.path.join(args.save_path, 'gantt_chart.jpg'), format='jpg')

    alert_curve, composite_score = engine.factor_analysis()
    alert_timestamps = engine.df[alert_curve == 1].index
    pd.Series(alert_timestamps).to_csv(os.path.join(args.save_path, 'alert_timestamps.csv'), index=False)


    

if __name__ == "__main__":
    args = parse_args()
    main(args)





