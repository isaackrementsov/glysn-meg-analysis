#! /usr/bin/bash
#SBATCH --job-name=story_decoder
#SBATCH --output=story_decoder.%j.out
#SBATCH --error=story_decoder.%j.err
#SBATCH --time=8:00:00
#SBATCH -p normal
#SBATCH -c 32
#SBATCH --mem=128GB
ml py-pandas/2.0.1_py39 
ml py-scikit-learn/1.3.2_py39 
pip3 install mne
python3 story_decoder.py NOUN VERB PRON ADP
