import uuid
import json
import os
import argparse


parser = argparse.ArgumentParser(
    description="Rename CFD files using UUIDs and store Re values in JSON"
)

parser.add_argument("--problem_folder", required=True)
parser.add_argument("--json_filename", required=True)

args = parser.parse_args()

# Load existing JSON if it exists
if os.path.exists(args.json_filename):
    with open(args.json_filename, "r") as file:
        data = json.load(file)
else:
    data = {}

for filename in os.listdir(args.problem_folder):

    # Only process CSV files with Re_ prefix
    if not filename.startswith("Re_") or not filename.endswith(".csv"):
        continue

    # Extract Re from filename
    re = float(filename.split("_")[-1].replace(".csv", ""))

    # Generate UUID
    file_uuid = str(uuid.uuid4())

    old_path = os.path.join(args.problem_folder, filename)
    new_filename = file_uuid + ".csv"
    new_path = os.path.join(args.problem_folder, new_filename)

    # Rename file
    os.rename(old_path, new_path)

    # Store mapping
    data[file_uuid] = re


with open(args.json_filename, "w") as file:
    json.dump(data, file, indent=4)