"""
run_docker.py
=============
Builds the Docker image and runs the full ingestion + scoring pipeline locally.

Usage:
    python tools/run_docker.py [--submission-dir solution]
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_TAG  = "eurozone-gdp-nowcasting:latest"


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"[run_docker] Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", default="solution")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    # Build image
    run(["docker", "build", "-f", "tools/Dockerfile", "-t", IMAGE_TAG, "."])

    # Ingestion
    run([
        "docker", "run", "--rm",
        "-v", f"{REPO_ROOT}/dev_phase:/app/dev_phase",
        "-v", f"{REPO_ROOT}/{args.submission_dir}:/app/submission",
        "-v", f"{REPO_ROOT}/ingestion_res:/app/ingestion_res",
        IMAGE_TAG,
        "python", "ingestion_program/ingestion.py",
        "--data-dir",       "dev_phase/input_data",
        "--output-dir",     "ingestion_res",
        "--submission-dir", "submission",
    ])

    # Scoring
    run([
        "docker", "run", "--rm",
        "-v", f"{REPO_ROOT}/dev_phase:/app/dev_phase",
        "-v", f"{REPO_ROOT}/ingestion_res:/app/ingestion_res",
        "-v", f"{REPO_ROOT}/scoring_res:/app/scoring_res",
        IMAGE_TAG,
        "python", "scoring_program/scoring.py",
        "--reference-dir",  "dev_phase/reference_data",
        "--prediction-dir", "ingestion_res",
        "--output-dir",     "scoring_res",
    ])

    print("\n[run_docker] Done. Check scoring_res/scores.json for results.")


if __name__ == "__main__":
    main()
