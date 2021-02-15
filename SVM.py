"""
Support Vector Machines
    Hyperplane is a flat affine subspace of hyperplane dimension n-1
    1-d hyperplane is a single point
    2-d hyperplane is a line
    3-d hyperplane is flat plane
    use hyperplanes to create a separation between classes

New points are classified based on what side of the point or hyperplane they land on
How to choose where to put the hyperplane (point, line, flat plane)
    use separator that maximizes the margins between the classes
    maximal margin classifier

when classes are not perfectly separable
    create ranges for classification
    use bias-variance trade off depending where to place the separator
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC

# read in csv file
df = pd.read_csv('UNZIP_ME_FOR_NOTEBOOKS_V3/DATA/mouse_viral_study.csv')

# create hyperplane and visualize dataset
sns.scatterplot(x='Med_1_mL', y='Med_2_mL', hue='Virus Present', data=df)
x = np.linspace(0,10,100)
m = -1
b =11
y = m*x+b
plt.plot(x,y,'black')
#plt.show()

