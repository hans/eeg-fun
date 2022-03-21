"""
Prepare CDR-format data for Broderick et al. 2018 dataset.
"""

from argparse import ArgumentParser
from pathlib import Path

from mfn400.adapters.broderick2018 import BroderickDatasetAdapter


def main(args):
    dataset = BroderickDatasetAdapter(args.eeg_dir, args.stim_path)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("eeg_dir", type=Path)
    p.add_argument("stim_path", type=Path)
