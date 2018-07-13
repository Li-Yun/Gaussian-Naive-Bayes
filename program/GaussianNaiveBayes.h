//
//  GaussianNaiveBayes.h
//  
//
//  Created by Wang, Li-Yun on 2/17/16.
//
//

#ifndef ____GaussianNaiveBayes__
#define ____GaussianNaiveBayes__

#include <iostream>
#include <vector>

using namespace std;

class GaussianNaiveBayes
{
    
private:
    
    vector < vector <double> > training_data;
    vector < vector <double> > testing_data;
    
public:
    
    GaussianNaiveBayes(){};
    ~GaussianNaiveBayes(){};
    
    // build probabilistic model
    void build_probabilistic_model(int input_mode);
    
    // build classification function
    void NB_classification(int input_mode);
    
};

#endif /* defined(____GaussianNaiveBayes__) */
