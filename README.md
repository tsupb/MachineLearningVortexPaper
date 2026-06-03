This code is the basis for the work done in the paper "Optical vortex classification via machine learning"

Two main scripts are included:
- main_TrainandTest.py: This script can be used to train a model on a given sample set and then benchmark the model according to several parameters
- main_Load.py: This script can be used to load a model trained with main_TrainandTest.py and the test it on a separate set of samples

All samples set should be placed in a folder "samples" in the project root directory. The sample folders name should uniquely identify the sample set and is used to identify the samples in the code. In the folder a single directory for each class (e.g. "+1" and "-1") should be contained, whith each directory containing the samples in a common image format (.png, .jpg) and with a unique name. Here numbers in ascending order are a good option as some programm outputs refer to them by numerical IDs in ascending order.
For model file outputs a diretory named "modelSaves" is expected in the project root directory.
Some outputs are saved as spreadsheets (.xlsx) are saved directly in the project root directory.

For the programm to run the following python packages need to be installed:
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
