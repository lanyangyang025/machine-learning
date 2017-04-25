Name: Chuanwenjie Wei, Yiqing Lu
NetID: cw64, yl128

Language: 
we use Matlab to load the data, and use keras (implemented in Python) to train the data.

List of dependencies:
We use keras to run the problem. It’s based on tensorflow.
The version of Python = 3.5.
The script of installation is shown in install.sh.

Run detail:
‘load_process.m’: We could use it to transfer the csv file to .mat. 
Although the dataset could be processed from the function in ‘load_process.m’, it seems that the dataset 
in mat file could be directly downloaded from the website http://ufldl.stanford.edu/housenumbers/, and the 
extra is difficult to be transferred into .mat file because of the lack of memory.
Before we run the code, we have to activate the tensorflow environment, and then launch jupyter 
notebook to run the code.

All the codes are changed from 'https://keras.io/'
To ensure that we could run the program correctly, we upload the dataset in './data' folder. What we need to do is to 
unzip the file and then run the .ipynb file.


'kera_final.ipynb': it’s the code of our final model. The accuracy should be 0.969(training data+extra data).
'kera_nin.ipynb': it’s the code of the NIN model. The accuracy should be 0.919(training data).
'kera_temp.ipynb': it’s the code of the VGG model. The accuracy should be 0.919(training data).

