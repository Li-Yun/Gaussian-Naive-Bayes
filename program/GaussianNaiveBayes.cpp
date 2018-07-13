//
//  GaussianNaiveBayes.cpp
//  
//
//  Created by Wang, Li-Yun on 2/17/16.
//
//

#include "GaussianNaiveBayes.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>

using namespace std;

// read testing data from the file
vector < vector <double> > read_testing_data()
{
    vector < vector <double> > output;
    
    // read testing data
	ifstream inputfile("../testing_data/testing_set.txt"); // open the file
	int row_number = 0;
	string line_1;
    
	while (getline(inputfile, line_1))
	{
		float temp_value;
		stringstream string_stream_1(line_1);
		output.push_back( vector<double>() );
        
		while (string_stream_1 >> temp_value)
        {
            output[row_number].push_back(temp_value);
        }
		
		row_number = row_number + 1;
	}
	inputfile.close();
    
    return output;
}

// read training data from the file
vector < vector <double> > read_training_data()
{
    vector < vector <double> > output;
    
    // read training data
	ifstream inputfile("../training_data/training_set.txt"); // open the file
	int row_number = 0;
	string line_1;
    
	while (getline(inputfile, line_1))
	{
		float temp_value;
		stringstream string_stream_1(line_1);
		output.push_back( vector<double>() );
        
		while (string_stream_1 >> temp_value)
        {
            output[row_number].push_back(temp_value);
        }
		
		row_number = row_number + 1;
	}
	inputfile.close();
    
    return output;
}

// training phrase in Gaussian Naive Bayes: build probabilistic model
void GaussianNaiveBayes::build_probabilistic_model(int input_mode)
{
    // setting variables
    int number_class_one = 0;
    int number_class_zero = 0;
    vector <double> class_one_mean;
    vector <double> class_zero_mean;
    vector <double> class_one_std;
    vector <double> class_zero_std;
    
    // read training data from the file
    training_data = read_training_data();
    
    // compute prior probability for each class
    for ( int i = 0 ; i < training_data.size() ; i++)
    {
        // count the number of "1" and "0"
        if (training_data[i][57] == 1.0)
        {
            number_class_one++;
        }
        else
        {
            number_class_zero++;
        }
    }
    
    // write prior probability to a file
    ofstream output_file("../prior_probability.txt");
    output_file << double(number_class_one) / training_data.size() << " " << double(number_class_zero) / training_data.size();
    output_file.close();
    
    // compute the mean value for each feature in each class
    for ( int feature_index = 0 ; feature_index < training_data[0].size() - 1 ; feature_index++)
    {
        double temp_class_one_value = 0.0;
        double temp_class_zero_value = 0.0;
        
        // go through all training examples to calculate the sum of class one and the sum of class zero
        for ( int data_index = 0 ; data_index < training_data.size() ; data_index++)
        {
            // check each example's label
            if (training_data[data_index][57] == 1.0)
            {
                temp_class_one_value += training_data[data_index][feature_index];
            }
            else
            {
                temp_class_zero_value += training_data[data_index][feature_index];
            }
        }
        
        // compute mean value for each feature
        class_one_mean.push_back( temp_class_one_value / double(number_class_one) );
        class_zero_mean.push_back( temp_class_zero_value / double(number_class_zero) );
    }
    
    // calculate stand deviation for each class
    for ( int feature_index = 0 ; feature_index < training_data[0].size() - 1 ; feature_index++)
    {
        double temp_value_class_one = 0.0;
        double temp_value_class_zero = 0.0;
        
        // go through all training examples
        for ( int data_index = 0 ; data_index < training_data.size() ; data_index++)
        {
            // check each example's label
            if (training_data[data_index][57] == 1.0)
            {
                temp_value_class_one = temp_value_class_one + ( ( training_data[data_index][feature_index] - class_one_mean[feature_index] ) * ( training_data[data_index][feature_index] - class_one_mean[feature_index] ) );
            }
            else
            {
                temp_value_class_zero = temp_value_class_zero + ( ( training_data[data_index][feature_index] - class_zero_mean[feature_index] ) * ( training_data[data_index][feature_index] - class_zero_mean[feature_index] ) );
            }
        }
        
        // check whether the temp value is zero or not. If the temp value is zero, adds a small value to the temp value
        if (temp_value_class_one == 0.0)
        {
            temp_value_class_one = temp_value_class_one + 1e-1;
        }
        if (temp_value_class_zero == 0.0)
        {
            temp_value_class_zero = temp_value_class_zero + 1e-1;
        }
        
        // compute the value of stand deviation for each class
        class_one_std.push_back( sqrt( temp_value_class_one / double(number_class_one) ) );
        class_zero_std.push_back( sqrt( temp_value_class_zero / double(number_class_zero) ) );
    }
    
    // according to mode value, there are two different modes: 0 means single Gaussian Distribution and 1 means multiple Gaussian Distributions
    if ( input_mode == 0 || input_mode == 1 )
    {
        // write training model to a file
        ofstream output_file_2("../training_model.txt");
        for (int feature_index = 0 ; feature_index < training_data[0].size() - 1 ; feature_index++)
        {
            output_file_2 << class_one_mean[feature_index] << " " << class_zero_mean[feature_index] << " " << class_one_std[feature_index] << " " << class_zero_std[feature_index] << "\n";
        }
        output_file_2.close();
    }
    else
    {
        cout << "Error Input." << endl;
    }
}

// Gaussian function
double gaussian_function(double feature_value, double mean_value, double std_value)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (feature_value - mean_value) / std_value;
    
    return inv_sqrt_2pi / std_value * exp(-0.5 * a * a);
}

// modify Gaussian function
double modify_gaussian_function(double feature_value, double mean_value, double std_value)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (feature_value - mean_value) / std_value;
    
    if (isinf( log( inv_sqrt_2pi / std_value ) ) == 1)
    {
        return 0.0 - (0.5 * a * a );
    }
    else
    {
        return log( inv_sqrt_2pi / std_value ) - (0.5 * a * a );
    }
}

// build confusion matrix
void build_confusion_matrix(float true_positive_number, float predicted_positive_number, float actual_positive_number, int total_examples)
{
    // create confusion matrix
    int confusion_matrix[2][2] = {0};
    confusion_matrix[0][0] = int(true_positive_number);
    confusion_matrix[0][1] = int(actual_positive_number) - int(true_positive_number);
    confusion_matrix[1][0] = int(predicted_positive_number) - int(true_positive_number);
    confusion_matrix[1][1] = total_examples - confusion_matrix[0][0] - confusion_matrix[0][1] - confusion_matrix[1][0];
    
    // display the confusion matrix
    cout << "            " << "  Spam " << "   Not_Spam " << endl;
    cout << "==================================" << endl;
    for (int i = 0 ; i < 2 ; i++)
    {
        if (i == 0)
        {
            cout << " Spam     | " << "  ";
        }
        else
        {
            cout << " Not_Spam | " << "  ";
        }
        
        for (int j = 0 ; j < 2 ; j++)
        {
            cout << confusion_matrix[i][j] << "       ";
        }
        cout << endl;
    }
    cout << endl;
    cout << " Aside: Row direction in confusion matrix means actural class, and " << endl;
    cout << " Column direction in confusion matrix means predicted class." << endl;
    cout << endl;
}

// classification function
void GaussianNaiveBayes::NB_classification(int input_mode)
{
    // read testing data from the file
    testing_data = read_testing_data();
    
    // setting variables
    double* predicted_vector = new double[testing_data.size()];
    double* prior_vector = new double[2];
    vector < vector <double> > model_information;
    int confusion_matrix[2][2] = {0};
    float accuracy = 0.0;
    float predicted_positive_number = 0.0;
    float true_positive_number = 0.0;
    float actual_positive_number = 0.0;
    
    // get prior probability
    ifstream inputfile("../prior_probability.txt"); // open the file
    string line_1;
    getline(inputfile, line_1);
    stringstream string_stream_1(line_1);
    string_stream_1 >> prior_vector[0];
    string_stream_1 >> prior_vector[1];
    inputfile.close();
    
    // if mode value is 0, there is only one Gaussian Distribution. If mode value is 1, there are multiple Gaussian functions in training data
    if ( input_mode == 0 )
    {
        // get training model
        ifstream inputfile_2("../training_model.txt"); // open the file
        int row_number_2 = 0;
        string line_2;
        
        while ( getline(inputfile_2, line_2) )
        {
            double temp_value;
            stringstream string_stream_2(line_2);
            model_information.push_back( vector<double>() );
            
            while (string_stream_2 >> temp_value)
            {
                model_information[row_number_2].push_back(temp_value);
            }
            
            row_number_2 = row_number_2 + 1;
        }
        inputfile_2.close();
        
        // classify each example in testing set
        for (int data_index = 0 ; data_index < testing_data.size() ; data_index++)
        {
            double class_one_value = log(prior_vector[0]);
            double class_zero_value = log(prior_vector[1]);
            
            for (int feature_index = 0 ; feature_index < testing_data[0].size() - 1 ; feature_index++)
            {
                class_one_value = class_one_value + modify_gaussian_function( testing_data[data_index][feature_index],  model_information[feature_index][0], model_information[feature_index][2] );
                class_zero_value = class_zero_value + modify_gaussian_function( testing_data[data_index][feature_index],  model_information[feature_index][1], model_information[feature_index][3] );
            }
            
            // check the sum value for each class, and determine predicted label
            if (class_one_value > class_zero_value)
            {
                predicted_vector[data_index] = 1.0;
            }
            else if (class_one_value < class_zero_value)
            {
                predicted_vector[data_index] = 0.0;
            }
            else
            {
                // generate a random value between 0 and 1
                double random_value = ((double) rand() / (double) (RAND_MAX));
                
                // apply random value to determine predicted results
                if (random_value >= 0.5)
                {
                    predicted_vector[data_index] = 1.0;
                }
                else
                {
                    predicted_vector[data_index] = 0.0;
                }
            }
        }
    }
    else if ( input_mode == 1 )
    {
        // get training model
        ifstream inputfile_2("../training_model.txt"); // open the file
        int row_number_2 = 0;
        string line_2;
        
        while ( getline(inputfile_2, line_2) )
        {
            double temp_value;
            stringstream string_stream_2(line_2);
            model_information.push_back( vector<double>() );
            
            while (string_stream_2 >> temp_value)
            {
                model_information[row_number_2].push_back(temp_value);
            }
            
            row_number_2 = row_number_2 + 1;
        }
        inputfile_2.close();
        
        // classify each example in testing set
        for (int data_index = 0 ; data_index < testing_data.size() ; data_index++)
        {
            double class_one_value = log(prior_vector[0]);
            double class_zero_value = log(prior_vector[1]);
            
            for (int feature_index = 0 ; feature_index < testing_data[0].size() - 1 ; feature_index++)
            {
                class_one_value = class_one_value + log( ( prior_vector[0] * 1.05 ) * pow( gaussian_function( testing_data[data_index][feature_index],  model_information[feature_index][0], model_information[feature_index][2] ), 0.62 ) );
                class_zero_value = class_zero_value + log( ( prior_vector[1] * 1.05 ) * pow( gaussian_function( testing_data[data_index][feature_index],  model_information[feature_index][1], model_information[feature_index][3] ), 0.62 ) );
            }
            
            // check the sum value for each class, and determine predicted label
            if (class_one_value > class_zero_value)
            {
                predicted_vector[data_index] = 1.0;
            }
            else if (class_one_value < class_zero_value)
            {
                predicted_vector[data_index] = 0.0;
            }
            else
            {
                // generate a random value between 0 and 1
                double random_value = ((double) rand() / (double) (RAND_MAX));
                
                // apply random value to determine predicted results
                if (random_value >= 0.5)
                {
                    predicted_vector[data_index] = 1.0;
                }
                else
                {
                    predicted_vector[data_index] = 0.0;
                }
            }
        }
    }
    
    // compute accuracy
    for (int data_index = 0 ; data_index < testing_data.size() ; data_index++)
    {
        if ( predicted_vector[data_index] == testing_data[data_index][57] )
        {
            accuracy++;
        }
    }
    accuracy = ( accuracy / testing_data.size() ) * 100.0;
    
    // compute precision
    for (int data_index = 0 ; data_index < testing_data.size() ; data_index++)
    {
        // coount the number of predicted positive
        if ( predicted_vector[data_index] == 1.0 )
        {
            predicted_positive_number = predicted_positive_number + 1.0;
        }
        
        // count true positive number
        if ( (predicted_vector[data_index] == 1.0) && (testing_data[data_index][57] == 1.0) && (predicted_vector[data_index] == testing_data[data_index][57]) )
        {
            true_positive_number = true_positive_number + 1.0;
        }
    }
    float precision = true_positive_number / predicted_positive_number * 100.0;
    
    // compute recall
    for (int data_index = 0 ; data_index < testing_data.size() ; data_index++)
    {
        // compute the number of positives in actual class
        if( testing_data[data_index][57] == 1.0 )
        {
            actual_positive_number = actual_positive_number + 1.0;
        }
    }
    float recall = true_positive_number / actual_positive_number * 100.0;
    
    // display accuracy, precision, and recall.
    cout << "Accuracy: " << accuracy << " %" << endl;
    cout << "Precision: " << precision << " %" << endl;
    cout << "Recall: " << recall << " %" << endl;
    
    // build confusion matrix
    cout << endl;
    cout << "Confusion Matrix: " << endl;
    build_confusion_matrix(true_positive_number, predicted_positive_number, actual_positive_number, testing_data.size());
}