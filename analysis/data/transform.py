"""
Transforms line transcriptions into time-range data with metadata about pod, and tape mentions.
"""
from analysis.src.data import Data
Data.load().save()
