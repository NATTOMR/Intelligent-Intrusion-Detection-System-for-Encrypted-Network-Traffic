import os
import urllib.request
import zipfile
import argparse

def download_file(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    # NOTE: CSE-CIC-IDS2018 is usually downloaded via AWS CLI or authenticated request 
    # to the Canadian Institute for Cybersecurity (CIC) website.
    # This is a placeholder for direct URL if you have one.
    # The official dataset requires registering at UNB's website.
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please download the CSE-CIC-IDS2018 dataset manually from: https://www.unb.ca/cic/datasets/ids-2018.html")
        print("Once downloaded, place the CSV files in the 'data/raw/' directory.")

def main():
    parser = argparse.ArgumentParser(description="Download CSE-CIC-IDS2018 dataset")
    parser.add_argument('--url', type=str, default="", help="Direct URL to dataset if available")
    args = parser.add_argument()
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    if args.url:
        dest_path = 'data/raw/CSE-CIC-IDS2018.zip'
        download_file(args.url, dest_path)
    else:
        print("No direct URL provided.")
        print("1. Go to: https://www.unb.ca/cic/datasets/ids-2018.html")
        print("2. Register and download the CSV version of the dataset.")
        print("3. Extract and place the CSV files into 'data/raw/' folder.")

if __name__ == "__main__":
    main()
