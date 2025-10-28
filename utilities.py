import requests
import zipfile
import os


DATASET_DOWNLOAD_LINK = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"


def download_and_extract_dataset():
    # Check if data exists
    if os.path.exists('data/diabetic_data.csv') and os.path.exists('data/IDS_mapping.csv'):
        print("Dataset already exists. Skipping download.")
        return
    
    # Download the dataset from the specified URL
    response = requests.get(DATASET_DOWNLOAD_LINK)
    if response.status_code == 200:
        with open('dataset.zip', 'wb') as f:
            f.write(response.content)
        
        with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
            
        os.remove('dataset.zip')
    else:
        raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    dataset = download_and_extract_dataset()