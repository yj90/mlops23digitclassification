System Requirements:
OS
Hardware

dummy commit to make the feature branch ahead of main branch 

e.g. 100 samples, 2 or binary class classification: image of carrot or turnip
      50 samples : carrots
      50 samples : turnips

      Data distrubution: balanced/uniform

      x amount of data for training 
      n-x amount of data for testing

      calculate some eval metric(train model (70 samples in training:35 carrots, 35 turnips), 30 samples in testing) == performance

In Practise:
     Train, Development, Test

     Train = Training the model(model type, model hyper params, model iteration)

     Dev = Selecting the model

     Test = Reporting the performance

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




