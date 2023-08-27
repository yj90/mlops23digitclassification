System Requirements:
OS
Hardware

How to set up:

install conda
conda create -n digits python=3.9
conda activate digits
pip install -r requirements.txt

How to Run:

Python exp.py


Feature:

Vary model hyper parameter

Meaning of Failure:

Poor performance metrics
Coding Runtime/Compile Error
The Model gavee wrong result on test sample during demo


Randomness In ML model:

1- When you create a split(means different training set & test set) so accuracy my have sliglty change

So in order to avoid this  we need to FREEZ the Data .i.e shuffle =False

2- Data Order (when learning is iterative means in DL model)

3- Weight initialization in the ML Model




