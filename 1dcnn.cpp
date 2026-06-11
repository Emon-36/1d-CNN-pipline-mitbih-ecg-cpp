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

void randomgen_1d(int size, float* output, float min_val = -0.1f, float max_val = 0.1f) {
    mt19937 gen(std::random_device{}());
   uniform_real_distribution<float> dis(min_val, max_val);

    for (int i = 0; i < size; ++i) {
        output[i] = dis(gen);
    }
}

vector<vector<vector<float>>> max_pooling1d(const vector<vector<vector<float>>>& input, int pool_size = 2, int stride = 2) {
    int num_samples = input.size();
    int input_length = input[0].size();
    int channels = input[0][0].size();
    int output_length = (input_length - pool_size) / stride + 1;

    vector<vector<vector<float>>> output(num_samples, vector<vector<float>>(output_length, vector<float>(channels, 0.0f)));

    for (int i = 0; i < num_samples; ++i) {
        for (int c = 0; c < channels; ++c) {
            for (int j = 0; j < output_length; ++j) {
                int start_idx = j * stride;
                float max_val = input[i][start_idx][c];
                
                for (int p = 1; p < pool_size; ++p) {
                    if (input[i][start_idx + p][c] > max_val) {
                        max_val = input[i][start_idx + p][c];
                    }
                }
                output[i][j][c] = max_val;
            }
        }
    }
    return output;
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
    float bias[32];

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

    float** kernel = new float*[32];
    for (int i = 0; i < 32; ++i) {
        kernel[i] = new float[5];
    }

    randomgen(32, 5, kernel, -0.5f, 0.5f);
    randomgen_1d(32, bias, -0.1f, 0.1f);


    //Training loop , avoid for inference, just comment it out.
    vector<vector<vector<float>>> feature(87554, vector<vector<float>>(183, vector<float>(32, 0.0f)));
    vector<vector<vector<float>>> pooled_feature(87554, vector<vector<float>>(91, vector<float>(32, 0.0f)));

    for (int epoch = 0; epoch < 1000; epoch++)
        {
            for (size_t i = 0; i < 87554; i++) 
                {
                  for (size_t j = 0; j < 183; j++) 
                     {
                       for (size_t k = 0; k < 32; k++) 
                          {
                            float temp = 0.0f;
                
                              for (size_t l = 0; l < 5; l++) 
                                 {
                                    temp += static_X[i][j + l] * kernel[k][l];
                                 }
                
               
                          feature[i][j][k] = temp + bias[k]; 
            }
        }
    }

   pooled_feature = max_pooling1d(feature,2,2);


}


    

    return 0;
}