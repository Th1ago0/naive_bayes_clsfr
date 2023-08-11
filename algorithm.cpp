//Classification model with naive bayes to classify digital signatures in c++;

// Warning supression;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-compare"

// Libraries;
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <math.h>

// NameSpace;
using namespace std;
using namespace std::chrono;

// Test data start index;
const int start_test = 900;

// The number of predictions that will be displayed;
const int num_of_iterations = 5;

// More control variables;
double calc_mean(vector<double> vect);
double calc_var(vector<double> vect);

// Methods;
void print2DVector(vector< vector<double> > vect);
vector< vector<double> > prior_prob(vector<double> vect);
vector< vector<double> > countclass(vector<double> vect);
vector< vector<double> > likelihood_doc_type(vector<double> class_vec, vector<double> doc_type, vector< vector<double> > count_class);
vector< vector<double> > likelihood_valid_cert(vector<double> class_vec, vector<double> valid_cert, vector< vector<double> > count_class);
vector< vector<double> > use_daysMean(vector<double> class_vec, vector<double> use_days, vector< vector<double> > count_class);
vector< vector<double> > use_daysVar(vector<double> class_vec, vector<double> use_days, vector< vector<double> > count_class);
vector< vector<double> > use_daysMetrics(vector< vector<double> > use_daysMean, vector< vector<double> > use_daysVar);
double calc_use_days_lh(double v, double mean_v, double var_v);

// The implementation method of the Naive Bayes theorem;
vector< vector<double> > calc_raw_prob(double doc_type, double valid_cert, double use_days, vector< vector<double> > 
prior, vector< vector<double> > lh_doc_type, vector< vector<double> > lh_valid_cert, vector< vector<double> > use_days_mean, vector< vector<double> > use_days_var);

// Evaluation methods;
vector< vector<double> > confusionMatrix(vector<double> matA, vector<double> matB);
double accuracy(vector<double> matA, vector<double> matB);

// Main function;
int main(){

    // clear the terminal
    system("clear");

    /* LOADING DATA */

    // Define the file path;
    string filePath = "./datasets/dataset.csv";

    // Object instace to receive the file content;
    ifstream input_file;

    // Open the file;
    input_file.open(filePath);

    // Check for erros;
    if(!input_file.is_open()){
       cout << "Failed to open file" << endl;
       return 0;
    }

    /* VARIABLES DECLARATION */

    // Double type scalable variables to handle the values of each column;
    double id_val;
    double doc_type_val;
    double class_val;
    double valid_cert_val;
    double use_days_val;

    // Vector type variables to all elements of each dataset column;
    vector<double> id;
    vector<double> doc_type;
    vector<double> class_vec;
    vector<double> valid_cert;
    vector<double> use_days;

    // Variable to store the file header;
    string header;

    // Variable to store each cell of the csv file;
    string cell;

    // Retrieves the header to disregard the line;
    getline(input_file, header);

    /* LOAD LOOP AND INITIAL CLEAN UP */

    while(input_file.good()){
    
        // Id column reading;
        getline(input_file, cell, ',');
    
        // Remove quotes;
        cell.erase(remove(cell.begin(), cell.end(), '\"'), cell.end());
    
        if(!cell.empty()){
        
            // Convert the id from string to double;
            id_val = stod(cell);
        
            // Append the value of x in the vector;
            id.push_back(id_val);
        
            // doc_type column reading;
            getline(input_file, cell, ',');
        
            // Convert the doc_type column to double;
            doc_type_val = stod(cell);
        
            // Apped the value of x in the vector;
            doc_type.push_back(doc_type_val);
        
            // class column reading;
            getline(input_file, cell, ',');
        
            // Convert the class column to double;
            class_val = stod(cell);
        
            // Apped the value of x in the vector;
            class_vec.push_back(class_val);
        
            // valid_cert column reading;
            getline(input_file, cell, ',');
        
            // Convert the valid_cert column to double;
            valid_cert_val = stod(cell);
        
            // Apped the value of x in the vector;
            valid_cert.push_back(valid_cert_val);
        
            // use_days column reading;
            getline(input_file, cell);
        
            // Convert the use_days column to double;
            use_days_val = stod(cell);
        
            // Apped the value of x in the vector;
            use_days.push_back(use_days_val);
        }
        else{
            break;
        }
    }

    /* SPLITING TRAIN AND TEST DATA */

    //Initiating runtime measurement;
    auto start = high_resolution_clock::now();
    cout << "Initiating the algorithm execution" << endl;

    // Training data for doc_type;
    vector<double> doc_typetrain_data;

    // Loads the vector;
    for(int i = 0; i < start_test; i++){
        doc_typetrain_data.push_back(doc_type.at(i));
    }

    // Training data for class_vec;
    vector<double> class_vectrain_data;

    // Loads the vector;
    for(int i = 0; i < start_test; i++){
        class_vectrain_data.push_back(class_vec.at(i));
    }

    // Training data for valid_cert;
    vector<double> valid_certtrain_data;

    // Loads the vector;
    for(int i = 0; i < start_test; i++){
        valid_certtrain_data.push_back(valid_cert.at(i));
    }

    // Training data for use_days;
    vector<double> use_daystrain_data;

    // Loads the vector;
    for(int i = 0; i < start_test; i++){
        use_daystrain_data.push_back(use_days.at(i));
    }

    // Testing data for use_days;
    vector<double> use_daystest_data;

    // Loads the vector;
    for(int i = start_test; i < id.size(); i++){
        use_daystest_data.push_back(use_days.at(i));
    }

    // Training data for type_doc;
    vector<double> doc_typetest_data;

    // Loads the vector;
    for(int i = start_test; i < id.size(); i++){
        doc_typetest_data.push_back(doc_type.at(i));
    }

    // Training data for class_vec;
    vector<double> class_vectest_data;

    // Loads the vector;
    for(int i = start_test; i < id.size(); i++){
        class_vectest_data.push_back(class_vec.at(i));
    }

    // Training data for valid_cert;
    vector<double> valid_certtest_data;

    // Loads the vector;
    for(int i = start_test; i < id.size(); i++){
        valid_certtest_data.push_back(valid_cert.at(i));
    }

    /* NAIVE BAYES ALGORITHM */

    cout << "Prior probability" << endl;

    // Prior probabilities;
    // Matriz 1x2;
    vector< vector<double> > prior = prior_prob(class_vectrain_data);
    cout << "Prior probabilities" << endl;
    print2DVector(prior);
    cout << endl;

    // Vector with class variable count;
    // Matriz 1x2;
    vector< vector<double> > count_class = countclass(class_vectrain_data);

    cout << "Conditional probability" << endl;

    // Likelihood to doc_type variable;
    // Matriz 2x3;
    vector< vector<double> > lh_doc_type = likelihood_doc_type(class_vectrain_data, doc_typetrain_data, count_class);
    cout << "\tdoc_type " << endl;
    print2DVector(lh_doc_type);
    cout << endl ;

    // Likelihood to valid_cert variable;
    vector< vector<double> > lh_valid_cert = likelihood_valid_cert(class_vectrain_data, valid_certtrain_data, count_class);
    cout << "\tvalid_cert " << endl;
    print2DVector(lh_valid_cert);
    cout << endl;

    // Mean and variance to use_days variable;
    //Matriz 1x2;
    vector< vector<double> > use_days_mean = use_daysMean(class_vectrain_data, use_daystrain_data, count_class);
    vector< vector<double> > use_days_var = use_daysVar(class_vectrain_data, use_daystrain_data, count_class);

    // use_days variable metrics;
    cout<< "\tuse_days " << endl;
    vector< vector<double> > use_days_metrics = use_daysMetrics(use_days_mean, use_days_var);
    print2DVector(use_days_metrics);
    cout << endl << endl ;

    // use_days variable Mean;
    cout << "use_days Mean: " << endl;
    print2DVector(use_days_mean);
    cout << endl;

    // use_days variable Variance;
    cout << "use_days Variance: " << endl;
    print2DVector(use_days_var);
    cout << endl;
    auto stop = high_resolution_clock::now();

    // Vector to the probabilities after the train;
    // Matriz 1x2;
    vector< vector<double> > raw(1, vector<double> (2, 0));

    cout << "Predicting probability on test data" << endl;

    // Print the first 5 predictions;
    for(int i = start_test; i < (start_test + num_of_iterations); i++){
    
        // Matriz 1x2;
        raw = calc_raw_prob(doc_type.at(i), valid_cert.at(i), use_days.at(i), prior, lh_doc_type, lh_valid_cert, use_days_mean, use_days_var);
        print2DVector(raw);
    }

    cout << endl << endl;

    // Logs the end of the algorithm;
    std::chrono::duration<double> elapsed_sec = stop - start;
    cout << "Execution time: " << elapsed_sec.count() << endl << endl;

    // Normalize the probabilities;
    vector<double> p1 (146);
    for(int i = 0; i < doc_typetest_data.size(); i++){
    
        // Matriz 1x2;
        raw = calc_raw_prob(doc_typetest_data.at(i), valid_certtest_data.at(i), use_daystest_data.at(i), prior, lh_doc_type, lh_valid_cert, use_days_mean, use_days_var);
    
        if((raw.at(0).at(0)) > 0.5){
            p1.at(i) = 0;
        }
        else if((raw.at(0).at(1)) > 0.5){
            p1.at(i) = 1;
        }
        else {}
    }

    // Confusion matrix;
    cout << "Confusion matrix" << endl;
    vector< vector<double> > table = confusionMatrix(p1, class_vectest_data);
    print2DVector(table);
    cout << endl;

    double acc = accuracy(p1, class_vectest_data);
    cout << "Acuraccy: " << acc << endl;

    // Sensivity TP / (TP + FN);
    double sensivity = (table.at(0).at(0) / (table.at(0).at(0) + table.at(0).at(1)));
    cout << "Sensivity: " << sensivity << endl;

    // Specificity TN / (TN + FP);
    double specificity = (table.at(1).at(1) / (table.at(1).at(1) + table.at(0).at(1)));
    cout << "Specificity: " << specificity << endl;

    return 0;
}

/* METHODS DECLARATION */

// Method to print the vector;
void print2DVector(vector< vector<double> > vect){
    for(int i = 0; i < vect.size(); i++){
        for(int j = 0; j < vect[i].size(); j++){
            cout << vect[i][j] << " " ;
        }
        cout << endl;
    }
}

// Method to calculate the prior probability on training data;
vector< vector<double> > prior_prob(vector<double> vect){

    // Matriz 1x2;
    vector< vector<double> > prior(1, vector<double> (2, 0));

    for(int i = 0; i < vect.size(); i++){
        if(vect.at(i) == 0){
            prior.at(0).at(0)++;
        }
        else{
            prior.at(0).at(1)++;
        }
    }
    prior.at(0).at(0) = prior.at(0).at(0) / vect.size();
    prior.at(0).at(1) = prior.at(0).at(1) / vect.size();

    return prior;
}

// Calculate class count (used to calculate probabilities of input variables );
vector< vector<double> > countclass(vector<double> vect){

    // Matriz 1x2;
    vector< vector<double> > prior(1, vector<double> (2, 0));

    for(int i = 0; i < vect.size(); i++){
        if(vect.at(i) == 0){
            prior.at(0).at(0)++;
        }
        else{
            prior.at(0).at(1)++;
        }
    }
    return prior;
}

// Calculates the probability of the training data of doc_type;
vector< vector<double> > likelihood_doc_type(vector<double> class_vec, vector<double> doc_type, vector< vector<double> > count_class){

    // Matriz 2x3;
    vector< vector<double> > lh_doc_type (2, vector<double>(3, 0));

    for(int i = 0; i < class_vec.size(); i++){
        if(class_vec.at(i) == 0){
            if(doc_type.at(i) == 1){
               lh_doc_type.at(0).at(0)++;
            }
            else if(doc_type.at(i) == 2){
                lh_doc_type.at(0).at(1)++;
            }
            else if(doc_type.at(i) == 3){
                lh_doc_type.at(0).at(2)++;
            }
            else {}
        }
        else if(class_vec.at(i) == 1){
            if(doc_type.at(i) == 1){
               lh_doc_type.at(1).at(0)++;
            }
            else if(doc_type.at(i) == 2){
                lh_doc_type.at(1).at(1)++;
            }
            else if(doc_type.at(i) == 3){
                lh_doc_type.at(1).at(2)++;
            }
            else {}
        }
        else{}
    }

    for(int i = 0; i < lh_doc_type.size(); i++){
        for(int j = 0; j < lh_doc_type[i].size(); j++){
            if(i == 0){
                lh_doc_type.at(i).at(j) = lh_doc_type.at(i).at(j) / count_class.at(0).at(0);
            }
            else if(i == 1){
                lh_doc_type.at(i).at(j) = lh_doc_type.at(i).at(j) / count_class.at(0).at(1);
            }
        }
    }

    return lh_doc_type;
}

// Calculates the probability of the training data of valid_cert;
vector< vector<double> > likelihood_valid_cert(vector<double> class_vec, vector<double> valid_cert, vector< vector<double> > count_class){

    // Matriz 2x2;
    vector< vector<double> > lh_valid_cert (2, vector<double>(2, 0));

    for(int i = 0; i < class_vec.size(); i++){
        if(class_vec.at(i) == 0){
            if(valid_cert.at(i) == 0){
               lh_valid_cert.at(0).at(0)++;
            }
            else if(valid_cert.at(i) == 1){
                lh_valid_cert.at(0).at(1)++;
            }
            else {}
        }
        else if(class_vec.at(i) == 1){
            if(valid_cert.at(i) == 0){
               lh_valid_cert.at(1).at(0)++;
            }
            else if(valid_cert.at(i) == 1){
                lh_valid_cert.at(1).at(1)++;
            }
            else {}
        }
        else{}
    }

    for(int i = 0; i < lh_valid_cert.size(); i++){
        for(int j = 0; j < lh_valid_cert[i].size(); j++){
            if(i == 0){
                lh_valid_cert.at(i).at(j) = lh_valid_cert.at(i).at(j) / count_class.at(0).at(0);
            }
            else if(i == 1){
                lh_valid_cert.at(i).at(j) = lh_valid_cert.at(i).at(j) / count_class.at(0).at(1);
            }
        }
    }

    return lh_valid_cert;
}

// Calculating the mean of use_days of the training data;
vector< vector<double> > use_daysMean(vector<double> class_vec, vector<double> use_days, vector< vector<double> > count_class){

    // Matriz 1x2;
    vector< vector<double> > mean(1, vector<double> (2, 0) );

    for(int i = 0; i < class_vec.size(); i++){
        if(class_vec.at(i) == 0 ){
            mean.at(0).at(0) += use_days.at(i);
        }
        else if(class_vec.at(i) == 1 ){
            mean.at(0).at(1) += use_days.at(i);
        }
        else{}
    }

    for(int i = 0; i < mean.size(); i++){
        for(int j = 0; j < mean[i].size(); j++){
            if(j == 0){
                mean.at(i).at(j) = mean.at(i).at(j) / count_class.at(0).at(0);
            }
            else if(j == 1){
                mean.at(i).at(j) = mean.at(i).at(j) / count_class.at(0).at(1);
            }
        }
    }

    return mean;
}

// Calculating the variance of use_days of the training data;
vector< vector<double> > use_daysVar(vector<double> class_vec, vector<double> use_days, vector< vector<double> > count_class){

    // Matriz 1x2;
    vector< vector<double> > var(1, vector<double> (2, 0));

    // Matriz 1x2;
    vector< vector<double> > mean = use_daysMean(class_vec, use_days, count_class);

    for(int i = 0; i < class_vec.size(); i++){
        if(class_vec.at(i) == 0){
            var.at(0).at(0) += pow((use_days.at(i) - mean.at(0).at(0)),2);
        }
        else if(class_vec.at(i) == 1){
            var.at(0).at(1) += pow((use_days.at(i) - mean.at(0).at(1)),2);
        }
        else {}
    }
    for(int i = 0; i < var.size(); i++){
        for(int j = 0; j < var[i].size(); j++){
            if(j == 0){
                var.at(i).at(j) = var.at(i).at(j) / (count_class.at(0).at(0) - 1);
            }
            else if(j == 1){
                var.at(i).at(j) = var.at(i).at(j) / (count_class.at(0).at(1) - 1);
            }
            else {}
        }
    }

    return var;
}

// Formats the metrics (mean and variance) from the use_days variable to a 2x2 matrix;
vector< vector<double> > use_daysMetrics(vector< vector<double> > use_daysMean, vector< vector<double> > use_daysVar){

    // Matriz 2x2;
    vector< vector<double> > metrics (2, vector<double>(2,0));

    metrics.at(0).at(0) = use_daysMean.at(0).at(0);
    metrics.at(0).at(1) = sqrt(use_daysVar.at(0).at(0));
    metrics.at(1).at(0) = use_daysMean.at(0).at(1);
    metrics.at(1).at(1) = sqrt(use_daysVar.at(0).at(1));

    return metrics;

}

// Calculates the probability of the use_days variable;
double calc_use_days_lh(double v, double mean_v, double var_v){
    double use_days_lh = 0;

    // Calculates the probability (likelihood);
    use_days_lh = (1 / (sqrt(2 * M_PI * var_v))) * exp( -(pow((v - mean_v), 2)) / (2 * var_v));

    return use_days_lh;
}

// Implementing Bayes theorem formula;
vector< vector<double> > calc_raw_prob(double doc_type, double valid_cert, double use_days, vector< vector<double> > prior, vector< vector<double> > lh_doc_type, vector< vector<double> > lh_valid_cert, vector< vector<double> > use_days_mean, vector< vector<double> > use_days_var) {

    // Matriz 1x2;
    vector<vector<double> > raw(1, vector<double> (2, 0)); 

    // Output variables probability;
    double num_s = lh_doc_type.at(1).at(doc_type - 1) * lh_valid_cert.at(1).at(valid_cert) * prior.at(0).at(1) * calc_use_days_lh(use_days, use_days_mean.at(0).at(1), use_days_var.at(0).at(1));

    // Input variables probability
    double num_p = lh_doc_type.at(0).at(doc_type - 1) * lh_valid_cert.at(0).at(valid_cert) * prior.at(0).at(0) * calc_use_days_lh(use_days, use_days_mean.at(0).at(0), use_days_var.at(0).at(0));

    // Denominador
    double denominator = lh_doc_type.at(1).at(doc_type - 1) * lh_valid_cert.at(1).at(valid_cert) * calc_use_days_lh(use_days, use_days_mean.at(0).at(1), use_days_var.at(0).at(1)) * prior.at(0).at(1) + lh_doc_type.at(0).at(doc_type - 1) * lh_valid_cert.at(0).at(valid_cert) * calc_use_days_lh(use_days, use_days_mean.at(0).at(0), use_days_var.at(0).at(0)) * prior.at(0).at(0);

    raw.at(0).at(1) = num_s / denominator;
    raw.at(0).at(0) = num_p / denominator;
    
    return raw;
}

// Returns TP FP FN TN;

vector< vector<double> > confusionMatrix(vector<double> matA, vector<double> matB){

    // Matriz 2x2;
    vector< vector<double> > table(2, vector<double> (2, 0));

    // matA = predicted;
    // matB = testing class;

    for(int i = 0; i < matA.size(); i++){

        // True negative
        if(matA.at(i) == 0 && matB.at(i) == 0){
            table.at(0).at(0) ++;
        }

        // True positive;
        else if(matA.at(i) == 1 && matB.at(i) == 1){
            table.at(1).at(1) ++;
        }

        // False positive;
        else if(matA.at(i) == 1 && matB.at(i) == 0){
            table.at(1).at(0) ++;
        }

        // False negative;
        else if(matA.at(i) == 0 && matB.at(i) == 1){
            table.at(0).at(1) ++;
        }
        else {}
    }

    return table;
}

// Return the accuracy;
double accuracy(vector<double> matA, vector<double> matB){
    int matARow = matA.size();
    int matBRow = matB.size();

    if(matARow != matBRow){
        cout << "Error calculating accuracy. Dimensions must be equal " << endl;
    }

    double sum = 0;

    for(int i = 0; i < matA.size(); i++){
        if(matA.at(i) == matB.at(i)){
           sum++;
        }
    }

    return sum / matA.size();
}

#pragma clang diagnostic pop