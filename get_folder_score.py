from get_score import compare_two_reports
import fire
import os
import csv

def run_main(
    folder_path: str,
    output_file: str="output.csv",
    w_mismatch: float=1.5,
    w_missing: float=2,
    w_surplus: float=1,
    plot_folder: str=None
    ):
    """
    Compare folder of reports and return file with all scores
    """
    for folder in os.listdir(folder_path):
        [final_report, prelim_report] = os.listdir(f"{folder_path}/{folder}/")

        score = compare_two_reports(f"{folder_path}/{folder}/{final_report}", f"{folder_path}/{folder}/{prelim_report}", w_mismatch, w_missing, w_surplus)
        
        file_exists = os.path.exists(output_file)

        # Open the CSV file in append mode
        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # If the file doesn't exist, write the header first
            if not file_exists:
                writer.writerow(['Report', 'Output'])  # Column names
            
            # Write the folder name and assistant output as a new row
            writer.writerow([folder, score])

def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()