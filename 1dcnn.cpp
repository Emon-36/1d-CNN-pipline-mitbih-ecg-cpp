#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include<random>

using namespace std;

// Compile-time configuration constants matching the original dataset size
const int ROWS = 87554; 
const int COLS = 187; 


float relu(float z) { 
    return (z > 0 ? z : 0); 
}

float cross_entropy_loss(float predicted, float actual) {
    float epsilon = 1e-15f;
    predicted = fmaxf(epsilon, fminf(1.0f - epsilon, predicted));
    return - (actual * logf(predicted) + (1.0f - actual) * logf(1.0f - predicted));
}

float conv1d(float input[ROWS][COLS], int kernel_size, int layer) {
    cout << "\n-> Conv1D Layer " << layer << " Processing......" << endl;
    cout << "first 3 values of input tensor: " << input[0][0] << ", " << input[0][1] << ", " << input[0][2] << endl;
    
    float dummy_output = input[0][0] * 0.5f; 
    return relu(dummy_output); 
}


void randomgen(int row, int column, float** output, float min_val = -1.0f, float max_val = 1.0f) {
    
    mt19937 gen(random_device{}());
    uniform_real_distribution<float> dis(min_val, max_val);

    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            output[i][j] = dis(gen);
        }
    }
}



struct Dataset {
    vector<vector<float>> X_array; 
    vector<int> y_array;           
};

// Standalone function to parse the raw CSV data sequentially
Dataset getRawDataset(const string& filePath) {
    Dataset data;
    ifstream file(filePath);

    if (!file.is_open()) {
        cerr << "Error: File Cannot be opened, Check the path......." << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string value;
        vector<float> rowFeatures;

        while (getline(ss, value, ',')) {
            rowFeatures.push_back(stof(value));
        }

        // Check if row matches the expected MIT-BIH dataset dimension (187 features + 1 label)
        if (rowFeatures.size() >= 188) {
            int label = static_cast<int>(rowFeatures.back());
            data.y_array.push_back(label);

            rowFeatures.pop_back(); 
            data.X_array.push_back(rowFeatures);
        }
    }
    file.close();

    return data;
}

int main() {
    string filePath = "mitbih_train.csv"; 
    
    // 1. Declare fixed-size static destination arrays safely outside the thread stack
    static float static_X[ROWS][COLS];
    static int static_y[ROWS];

    // 2. Fetch raw data from CSV pipeline
    Dataset dataset = getRawDataset(filePath);
    
    if (dataset.X_array.empty()) {
        cout << "No data loaded." << endl;
        return 1;
    }

    // 3. Directly copy data features and labels into the static structures
    for (int i = 0; i < ROWS; ++i) {
        static_y[i] = dataset.y_array[i];
        for (int j = 0; j < COLS; ++j) {
            static_X[i][j] = dataset.X_array[i][j];
        }
    }

    // 4. Free vector overhead allocations instantly
    dataset.X_array.clear();
    dataset.y_array.clear();

    // 5. Print out the raw static array components to confirm values
    cout << "--- Printing First 10 Elements From Raw Static Array Block ---" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << "Sample [" << i + 1 << "] -> Label: " << static_y[i] << " | Signal: ";
        for (int j = 0; j < 3; ++j) {
            cout << static_X[i][j] << " ";
        }
        cout << "..." << endl;
    }


    //Training loop , avoid for inference, just comment it out.

    for (int epoch = 0; epoch < 1000; epoch++)
    {
        
    }
    


    

    return 0;
}