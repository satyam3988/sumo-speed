import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_chart(directory_path):
    # Get the CSV file from the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print("No CSV files found in the provided directory.")
        sys.exit(1)
    elif len(csv_files) > 1:
        print("Multiple CSV files found. Please ensure there's only one CSV file in the directory.")
        sys.exit(1)

    csv_path = os.path.join(directory_path, csv_files[0])

    # Read the CSV file
    data = pd.read_csv(csv_path)

    # Select every 10th row using iloc
    data = data.iloc[::150, :]

    # Plotting the data
    plt.figure(figsize=(10,6))
    plt.plot(data['Steps'], data['system_mean_waiting_time'], label='System Mean Waiting Time', color='blue')
    plt.xlabel('Steps')
    plt.ylabel('System Mean Waiting Time')
    plt.title('System Mean Waiting Time vs Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to PNG file
    png_path = os.path.join(directory_path, "system_mean_waiting_time_vs_steps.png")
    plt.savefig(png_path, format='png')
    print(f"Plot saved to: {png_path}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the directory containing the CSV file as an argument.")
        sys.exit(1)

    directory_path = sys.argv[1]
    plot_chart(directory_path)
