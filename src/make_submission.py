"""
This script converts test_labels.json into submission.csv
python make_submission.py --json_path test_labels_naive_baseline.json
"""
import json
from pathlib import Path

input_folder = Path("labels") / Path("test")
output_folder = Path("submissions")

default_submission_input = Path("test_labels_naive_baseline.json")
default_submission_output = Path("submission_naive_baseline.csv")


def make_submission(
    json_path: Path = default_submission_input,
    output_path: Path = default_submission_output,
):
    """Create a submission CSV file from a label json file output"""
    with open(input_folder / json_path, "r") as file:
        test_labels = json.load(file)

    file = open(output_folder / output_path, "w")
    file.write("id,target_feature\n")
    for key, value in test_labels.items():
        u_id = [key + "_" + str(i) for i in range(len(value))]
        target = map(str, value)
        for row in zip(u_id, target):
            file.write(",".join(row))
            file.write("\n")
    file.close()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(make_submission)
