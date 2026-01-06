import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MinuteLocator, date2num
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

class GanttChart:
    def __init__(self, df, all_keys, colors):
        """
        Initialize GanttChart with data and colors.
        
        Args:
            df (pd.DataFrame): DataFrame containing the activity data
            all_keys (list): List of activity keys
            colors (list): List of color hex codes for each activity
        """
        self.df = df
        self.all_keys = all_keys
        self.colors = colors
        self.cmaps = [self._create_colormap(color) for color in colors]
        
    def _create_colormap(self, color):
        """
        Create a colormap from a single color.
        
        Args:
            color (str): Hex color code
            
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Colormap object
        """
        return LinearSegmentedColormap.from_list('custom', ['white', color])
    
    def create_chart(self):
        """
        Create and return a gantt chart figure.
        
        Returns:
            matplotlib.figure.Figure: The gantt chart figure
        """
        # Convert string timestamps to datetime objects
        self.df.index = pd.to_datetime(self.df.index, format='%m-%d %H:%M:%S')

        fig, ax = plt.subplots(figsize=(36, 9))
        row_positions = [0, 1, 2, 3, 4, 5, 6, 7]
        
        for y_pos, key in zip(row_positions, self.all_keys):
            for i, (date, row) in enumerate(self.df.iterrows()):
                if i+1==len(self.df):
                    break
                norm_value = (row[key] - self.df[key].min()) / (self.df[key].max() - self.df[key].min())
                color = self.cmaps[y_pos](norm_value)
                
                current_time = self.df.index[i]
                next_time = self.df.index[i+1]
                width = mdates.date2num(next_time) - mdates.date2num(current_time)
                ax.barh(y=y_pos, width=width, left=mdates.date2num(current_time), 
                        height=0.8, color=color, alpha=0.8, edgecolor='none')
        
        x_min = date2num(self.df.index.min())
        x_max = date2num(self.df.index.max())
        # Ensure no padding is added
        ax.set_xlim(x_min, x_max)
        ax.autoscale(enable=False)  # Prevent auto-scaling
            
        # Customize the plot
        ax.set_yticks(row_positions)
        ax.set_yticklabels(self.all_keys, rotation=60)
        ax.set_ylim(-0.5, 8.5)  # Tight bounds around the two bars

        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(MinuteLocator(interval=30))
        plt.xticks(rotation=45)
        plt.xlabel('Time')

        # Remove unnecessary spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

        # plt.title('Combined Activity Timeline')
        plt.tight_layout()
        plt.subplots_adjust(left=0.03)
        
        return fig
