import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (10,10));

    # Skewed feature plotting
    for i, feature in enumerate(['Age','Fare']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 500))
        ax.set_yticks([0, 100,200,300,400,500])
        ax.set_yticklabels([0, 100,200,300,400,'>500'])


    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous TITANIC Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous TITANIC Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()

