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
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(comp_name + "/csvfile")


download_and_extract_kaggle_competition(comp_name)
