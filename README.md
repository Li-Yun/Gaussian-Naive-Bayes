# Gaussian-Naive-Bayes
This progrgam is a spambase data application througg Gaussian Naïve Bayes <br />
based on the UCI ML repository (https://archive.ics.uci.edu/ml/datasets/Spambase). <br />
By giving the original dataset, a training and testing dataset is yield by splitting <br />
the half dataset with about 40% spam and 60% non-spam. The program is then to build one <br />
or multiple Gussian distributions in the training phase given the training dataset. <br />
To classify new instances, the program utilizes Naive Bayes and Bayes Rules to <br />
determine what the class of new instances is.

# Usage

To compile the program, please follow this command: <br />

g++ -std=c++11 main.cpp GaussianNaiveBayes.cpp -o file_name 

For example: 

g++ -std=c++11 main.cpp GaussianNaiveBayes.cpp -o output

To start training all perceptron classifiers, please use the following command: <br />

./output naive_bayes_training [0 / 1]

To test new instances in a testing dataset, please run the following command: <br />

./outout classify_data [0 / 1]

NOTE: <br />
The second parameter in the command is to choose different models. Value 0 represents a regular model, <br />
which is only one Gaussian distrubtion. By contrast, Value 1 means that the program uses multiple Gaussian <br />
distrubiotns to fit the training dataset.

# Have Fun!!
