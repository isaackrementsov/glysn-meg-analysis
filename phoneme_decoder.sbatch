#! /usr/bin/bash
#SBATCH --job-name=phoneme_decoder
#SBATCH --output=phoneme_decoder.%j.out
#SBATCH --error=phoneme_decoder.%j.err
#SBATCH --time=2:00:00
#SBATCH -p normal
#SBATCH -c 32
#SBATCH --mem=128GB
ml py-pandas/2.0.1_py39 
ml py-scikit-learn/1.3.2_py39 
pip3 install mne
python3 phoneme_decoder.py
