
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

my_file = Path('./Startups.csv')
if my_file.is_file():
    print ('File exists')
else:
    print ('File does not exist')
