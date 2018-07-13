#include <iostream>
#include "GaussianNaiveBayes.h"

using namespace std;

// main function
int main(int argc, char* argv[])
{
    // check argument format
    if (argc <= 1)
    {
        cout << "Argument Error." << endl;
        cout << "The program only has two arguments: ./a.out and naive_bayes_training/classify_data" << endl;
    }
    else
    {
        // get character array, and convert to string
        string input_string(argv[1]);
        
        // choose different learning model
        int model_chose = atoi(argv[2]);
        
        // create Gaussian Naive Bayes Classifer object
        GaussianNaiveBayes* GNB_classifier = new GaussianNaiveBayes();
        
        
        // check input string
        if ( input_string.compare("naive_bayes_training") == 0 )
        {
            GNB_classifier -> build_probabilistic_model(model_chose);
        }
        else if ( input_string.compare("classify_data") == 0 )
        {
            GNB_classifier -> NB_classification(model_chose);
        }
        else
        {
            cout << "Error argument." << endl;
            cout << "Argument Format: ./a.out naive_bayes_training/classify_data 0/1" << endl;
        }
    }
    
    return 0;
}