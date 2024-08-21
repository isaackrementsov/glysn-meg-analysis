#! /usr/bin/bash
#SBATCH --job-name=meg_decoding
#SBATCH --output=meg_decoding.%j.out
#SBATCH --error=meg_decoding.%j.err
#SBATCH --time=05:00:00
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=32GB

python3 get_story_decoding_scores.py
