# IAdaBoost-Experiment
Experiment on IAdaBoost Method for Learning from Imbalanced Class Distributions.
IAdaBoost: An Improved Method for Class Imbalance Learning
IAdaBoost is an improved method for addressing class imbalance in machine learning, particularly effective for highly or extremely imbalanced datasets. This repository contains the implementation of the IAdaBoost classifier and experimental data related to class imbalance.

After performing hyperparameter optimization, the model is evaluated based on the G-Mean metric. The evaluation results are from 10 random experiments on the test dataset.

```sh
python IAdaBoost.py
```

Requirements：
All experiments were performed on a desktop computer with the following configuration:

CPU: Intel i5-12400F
RAM: 16 GB
Operating System: Windows 11
Python: 3.11

Packages Used：
Package             Version
------------------- -----------
colorama            0.4.6
contourpy           1.3.0
cycler              0.12.1
et_xmlfile          2.0.0
fonttools           4.54.1
imbalanced-ensemble 0.1.7
imbalanced-learn    0.12.4
imblearn            0.0
joblib              1.4.2
kiwisolver          1.4.7
matplotlib          3.9.2
numpy               1.23.5
opencv-python       4.10.0.84
openpyxl            3.1.5
packaging           24.2
pandas              2.2.3
pillow              11.0.0
pip                 23.2.1
pyparsing           3.2.0
python-dateutil     2.9.0.post0
pytz                2024.2
pywin32             308
scikit-learn        1.1.3
scipy               1.14.1
seaborn             0.13.2
setuptools          65.5.0
six                 1.16.0
svgwrite            1.4.3
threadpoolctl       3.5.0
tqdm                4.67.0
tzdata              2024.2
