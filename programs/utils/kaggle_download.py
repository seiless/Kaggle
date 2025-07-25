import os
import json
import subprocess
import zipfile

two_levels_up = os.path.abspath(os.path.join(os.getcwd(), ".."))
json_path = os.path.join(two_levels_up, "kaggle.json")
with open(json_path, "r") as f:
    json_data = json.load(f)

comp_name = input()

os.environ["KAGGLE_USERNAME"] = json_data["username"]
os.environ["KAGGLE_KEY"] = json_data["key"]


def download_and_extract_kaggle_competition(competition_name: str):
    target_dir = f"./{competition_name}"
    os.makedirs(f"./{competition_name}", exist_ok=True)
    zip_path = os.path.join(target_dir, f"{competition_name}.zip")

    result = subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition_name,
            "-p",
            target_dir,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print(
            "Download failed. Please make sure Kaggle API is authenticated and the competition name is correct."
        )
        return

    # Extract ZIP
    print("Attempting to extract files...")
    try:
        if not os.path.exists(zip_path):
            print(f"Error: Cannot find the file to extract. Path: {zip_path}")
            print("Please check if the download completed successfully.")
            return

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            extract_path = os.path.join(target_dir, "data")
            zip_ref.extractall(extract_path)

        os.remove(zip_path)
        print(f"Extraction complete. Files are saved in: '{extract_path}'")

    except zipfile.BadZipFile:
        print(f"Error: The file '{zip_path}' is corrupted or not a valid zip file.")
        print("Please delete the file and try again.")
    except Exception as e:
        print(f"An unknown error occurred during extraction: {e}")


download_and_extract_kaggle_competition(comp_name)
