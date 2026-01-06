import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_skeleton_data(pt_path, csv_path):
    try:
        # Load the PyTorch file
        skeleton_data = torch.load(pt_path)
        print("Skeleton data type:", type(skeleton_data))
        print("Skeleton data size:", skeleton_data.size())
        
        # Load the CSV file
        csv_data = pd.read_csv(csv_path)
        print("\nCSV data shape:", csv_data.shape)
        print("\nFirst few timestamps from CSV:")
        print(csv_data.iloc[:5, 0])  # Show first 5 timestamps
        
        # Verify the lengths match
        if len(csv_data) == skeleton_data.size(0):
            print("\n✓ Number of frames match between PT and CSV files")
        else:
            print("\n⚠ Warning: Number of frames don't match!")
            print(f"PT file frames: {skeleton_data.size(0)}")
            print(f"CSV file frames: {len(csv_data)}")
        
        # Create 3D plot of first keypoint trajectory
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get timestamps and normalize them for better visualization
        timestamps = csv_data.iloc[:, 0].values
        timestamps_normalized = (timestamps - timestamps[0]) / 1000  # Convert to seconds from start
        
        # Get x and y coordinates of first keypoint
        x_coords = skeleton_data[:, 0, 0].numpy()  # First keypoint, x coordinate
        y_coords = skeleton_data[:, 0, 1].numpy()  # First keypoint, y coordinate
        
        # Create the 3D plot
        scatter = ax.scatter(x_coords, y_coords, timestamps_normalized, 
                           c=timestamps_normalized, cmap='viridis',
                           s=20, alpha=0.6)
        
        # Connect points with lines
        ax.plot(x_coords, y_coords, timestamps_normalized, 'gray', alpha=0.3)
        
        # Add labels and title
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Time (seconds)')
        ax.set_title('Trajectory of First Keypoint Over Time')
        
        # Add colorbar
        plt.colorbar(scatter, label='Time (seconds)')
        
        # Adjust the viewing angle to make x-axis horizontal
        ax.view_init(elev=0, azim=0)  # Changed to make x-axis horizontal
        
        # Show the plot
        plt.show()
            
    except Exception as e:
        print(f"Error reading files: {e}")

if __name__ == "__main__":
    pt_path = "/Users/kevynzhang/Downloads/skeleton/cam_10/giver_1_skeleton.pt"
    csv_path = "/Users/kevynzhang/Downloads/skeleton/cam_10/giver_1.csv"
    read_skeleton_data(pt_path, csv_path)
