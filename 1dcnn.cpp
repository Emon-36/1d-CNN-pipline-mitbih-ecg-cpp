#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>

using namespace std;

const int ROWS = 87554; 
const int COLS = 187; 

float relu(float z) { 
    return (z > 0 ? z : 0); 
}

float relu_derivative(float z) {
    return (z > 0 ? 1.0f : 0.0f);
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

vector<float> softmax(const vector<float>& logits) {
    vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] /= sum;
    }
    return probs;
}

struct Dataset {
    vector<vector<float>> X_array; 
    vector<int> y_array;           
};

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
    
    static float static_X[ROWS][COLS];
    static int static_y[ROWS];
    
    // Layer 1 Accumulators
    float bias[32];
    float d_bias[32] = {0.0f};

    Dataset dataset = getRawDataset(filePath);
    
    if (dataset.X_array.empty()) {
        cout << "No data loaded." << endl;
        return 1;
    }

    for (int i = 0; i < ROWS; ++i) {
        static_y[i] = dataset.y_array[i];
        for (int j = 0; j < COLS; ++j) {
            static_X[i][j] = dataset.X_array[i][j];
        }
    }

    dataset.X_array.clear();
    dataset.y_array.clear();

    cout << "--- Printing First 10 Elements From Raw Static Array Block ---" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << "Sample [" << i + 1 << "] -> Label: " << static_y[i] << " | Signal: ";
        for (int j = 0; j < 3; ++j) {
            cout << static_X[i][j] << " ";
        }
        cout << "..." << endl;
    }

    // Layer 1 Kernels and Gradients Allocation
    float** kernel = new float*[32];
    float** d_kernel = new float*[32];
    for (int i = 0; i < 32; ++i) {
        kernel[i] = new float[5];
        d_kernel[i] = new float[5]();
    }

    randomgen(32, 5, kernel, -0.5f, 0.5f);
    randomgen_1d(32, bias, -0.1f, 0.1f);

    const int OUT_CHANNELS_L2 = 64;
    const int KERNEL_SIZE_L2 = 5;
    const int IN_CHANNELS_L2 = 32;
    const int OUT_LENGTH_L2 = 87; 

    vector<vector<vector<float>>> feature(87554, vector<vector<float>>(183, vector<float>(32, 0.0f)));
    vector<vector<vector<float>>> pooled_feature(87554, vector<vector<float>>(91, vector<float>(32, 0.0f)));
    vector<vector<vector<float>>> feature_L2(87554, vector<vector<float>>(OUT_LENGTH_L2, vector<float>(OUT_CHANNELS_L2, 0.0f)));
    vector<vector<vector<float>>> pooled_feature_L2(87554, vector<vector<float>>(43, vector<float>(64, 0.0f)));

    // Layer 2 Accumulators
    float bias_L2[64];
    float d_bias_L2[64] = {0.0f};
    randomgen_1d(64, bias_L2, -0.1f, 0.1f);

    // Layer 2 Kernels and Gradients Allocation
    float*** kernel_L2 = new float**[OUT_CHANNELS_L2];
    float*** d_kernel_L2 = new float**[OUT_CHANNELS_L2];
    for (int k = 0; k < OUT_CHANNELS_L2; ++k) {
        kernel_L2[k] = new float*[KERNEL_SIZE_L2];
        d_kernel_L2[k] = new float*[KERNEL_SIZE_L2];
        for (int l = 0; l < KERNEL_SIZE_L2; ++l) {
            kernel_L2[k][l] = new float[IN_CHANNELS_L2];
            d_kernel_L2[k][l] = new float[IN_CHANNELS_L2]();
            for (int c = 0; c < IN_CHANNELS_L2; ++c) {
                kernel_L2[k][l][c] = ((float)rand() / RAND_MAX) * 1.0f - 0.5f;
            }
        }
    }

    const int FLATTEN_SIZE = 43 * 64; 
    const int NUM_CLASSES = 5;       
    const float LEARNING_RATE = 0.05f;
    const int BATCH_SIZE = 100;

    // FC Layer Allocation
    float** fc_weights = new float*[NUM_CLASSES];
    float** d_fc_weights = new float*[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; ++i) {
        fc_weights[i] = new float[FLATTEN_SIZE];
        d_fc_weights[i] = new float[FLATTEN_SIZE]();
    }
    randomgen(NUM_CLASSES, FLATTEN_SIZE, fc_weights, -0.05f, 0.05f);

    float fc_bias[NUM_CLASSES];
    float d_fc_bias[NUM_CLASSES] = {0.0f};
    randomgen_1d(NUM_CLASSES, fc_bias, -0.01f, 0.01f);

    // --- TRAINING LOOP (ALL LAYERS UPDATING) ---
    for (int epoch = 0; epoch < 1000; epoch++)
    {
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < 87554; i++) 
        {
            // 1. Forward Pass: Conv1
            for (size_t j = 0; j < 183; j++) 
            {
                for (size_t k = 0; k < 32; k++) 
                {
                    float temp = 0.0f;
                    for (size_t l = 0; l < 5; l++) 
                    {
                        temp += static_X[i][j + l] * kernel[k][l];
                    }
                    feature[i][j][k] = relu(temp + bias[k]); 
                }
            }
        }

        pooled_feature = max_pooling1d(feature, 2, 2);

        for (size_t i = 0; i < 87554; i++) 
        {
            // Forward Pass: Conv2
            for (size_t j = 0; j < OUT_LENGTH_L2; j++) 
            {
                for (size_t k = 0; k < OUT_CHANNELS_L2; k++) 
                {
                    float temp = 0.0f;
                    for (size_t l = 0; l < KERNEL_SIZE_L2; l++) 
                    {
                        for (size_t c = 0; c < IN_CHANNELS_L2; c++) 
                        {
                            temp += pooled_feature[i][j + l][c] * kernel_L2[k][l][c];
                        }
                    }
                    feature_L2[i][j][k] = relu(temp + bias_L2[k]);
                }
            }
        }

        pooled_feature_L2 = max_pooling1d(feature_L2, 2, 2);

        for (size_t i = 0; i < 87554; i++)
        {
            // Flatten
            vector<float> flattened(FLATTEN_SIZE);
            int index = 0;
            for (int j = 0; j < 43; ++j) {
                for (int k = 0; k < 64; ++k) {
                    flattened[index++] = pooled_feature_L2[i][j][k];
                }
            }

            // Forward Pass: Dense (FC)
            vector<float> logits(NUM_CLASSES, 0.0f);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float temp = 0.0f;
                for (int f = 0; f < FLATTEN_SIZE; ++f) {
                    temp += flattened[f] * fc_weights[c][f];
                }
                logits[c] = temp + fc_bias[c];
            }

            vector<float> probs = softmax(logits);

            int target_class = static_y[i];
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float actual = (c == target_class) ? 1.0f : 0.0f;
                epoch_loss += cross_entropy_loss(probs[c], actual);
            }

            // 2. BACKPROPAGATION (ALL LAYERS)
            vector<float> d_logits(NUM_CLASSES);
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float actual = (c == target_class) ? 1.0f : 0.0f;
                d_logits[c] = probs[c] - actual;
            }

            // Gradients for FC
            for (int c = 0; c < NUM_CLASSES; ++c) {
                d_fc_bias[c] += d_logits[c];
                for (int f = 0; f < FLATTEN_SIZE; ++f) {
                    d_fc_weights[c][f] += d_logits[c] * flattened[f];
                }
            }

            // Backprop to Flattened
            vector<float> d_flattened(FLATTEN_SIZE, 0.0f);
            for (int f = 0; f < FLATTEN_SIZE; ++f) {
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    d_flattened[f] += d_logits[c] * fc_weights[c][f];
                }
            }

            // Unflatten to MaxPool2 output grad
            vector<vector<float>> d_pooled_feature_L2(43, vector<float>(64, 0.0f));
            index = 0;
            for (int j = 0; j < 43; ++j) {
                for (int k = 0; k < 64; ++k) {
                    d_pooled_feature_L2[j][k] = d_flattened[index++];
                }
            }

            // Backprop MaxPool2 to Conv2
            vector<vector<float>> d_feature_L2(OUT_LENGTH_L2, vector<float>(OUT_CHANNELS_L2, 0.0f));
            for (int k = 0; k < OUT_CHANNELS_L2; ++k) {
                for (int j = 0; j < 43; ++j) {
                    int start_idx = j * 2;
                    int max_idx = start_idx;
                    float max_val = feature_L2[i][start_idx][k];
                    if (feature_L2[i][start_idx + 1][k] > max_val) {
                        max_idx = start_idx + 1;
                    }
                    d_feature_L2[max_idx][k] = d_pooled_feature_L2[j][k] * relu_derivative(feature_L2[i][max_idx][k]);
                }
            }

            // Gradients for Conv2 weights/bias & Backprop to MaxPool1
            vector<vector<float>> d_pooled_feature(91, vector<float>(32, 0.0f));
            for (int j = 0; j < OUT_LENGTH_L2; ++j) {
                for (int k = 0; k < OUT_CHANNELS_L2; ++k) {
                    float grad_out = d_feature_L2[j][k];
                    d_bias_L2[k] += grad_out; // Update Layer 2 Bias Grad
                    for (int l = 0; l < KERNEL_SIZE_L2; ++l) {
                        for (int c = 0; c < IN_CHANNELS_L2; ++c) {
                            d_kernel_L2[k][l][c] += pooled_feature[i][j + l][c] * grad_out; // Update Layer 2 Weight Grad
                            d_pooled_feature[j + l][c] += kernel_L2[k][l][c] * grad_out;
                        }
                    }
                }
            }

            // Backprop MaxPool1 to Conv1
            vector<vector<float>> d_feature(183, vector<float>(32, 0.0f));
            for (int k = 0; k < 32; ++k) {
                for (int j = 0; j < 91; ++j) {
                    int start_idx = j * 2;
                    int max_idx = start_idx;
                    float max_val = feature[i][start_idx][k];
                    if (feature[i][start_idx + 1][k] > max_val) {
                        max_idx = start_idx + 1;
                    }
                    d_feature[max_idx][k] = d_pooled_feature[j][k] * relu_derivative(feature[i][max_idx][k]);
                }
            }

            // Gradients for Conv1 weights/bias
            for (int j = 0; j < 183; ++j) {
                for (int k = 0; k < 32; ++k) {
                    float grad_out = d_feature[j][k];
                    d_bias[k] += grad_out; // Update Layer 1 Bias Grad
                    for (int l = 0; l < 5; ++l) {
                        d_kernel[k][l] += static_X[i][j + l] * grad_out; // Update Layer 1 Weight Grad
                    }
                }
            }

            // 3. WEIGHT UPDATES (BATCH STEP)
            if ((i + 1) % BATCH_SIZE == 0 || (i + 1) == 87554) {
                int current_batch_size = ((i + 1) % BATCH_SIZE == 0) ? BATCH_SIZE : (87554 % BATCH_SIZE);

                // FC Layer Update
                for (int c = 0; c < NUM_CLASSES; ++c) {
                    fc_bias[c] -= LEARNING_RATE * (d_fc_bias[c] / current_batch_size);
                    d_fc_bias[c] = 0.0f;
                    for (int f = 0; f < FLATTEN_SIZE; ++f) {
                        fc_weights[c][f] -= LEARNING_RATE * (d_fc_weights[c][f] / current_batch_size);
                        d_fc_weights[c][f] = 0.0f;
                    }
                }

                // Layer 2 Update
                for (int k = 0; k < OUT_CHANNELS_L2; ++k) {
                    bias_L2[k] -= LEARNING_RATE * (d_bias_L2[k] / current_batch_size);
                    d_bias_L2[k] = 0.0f;
                    for (int l = 0; l < KERNEL_SIZE_L2; ++l) {
                        for (int c = 0; c < IN_CHANNELS_L2; ++c) {
                            kernel_L2[k][l][c] -= LEARNING_RATE * (d_kernel_L2[k][l][c] / current_batch_size);
                            d_kernel_L2[k][l][c] = 0.0f;
                        }
                    }
                }

                // Layer 1 Update
                for (int k = 0; k < 32; ++k) {
                    bias[k] -= LEARNING_RATE * (d_bias[k] / current_batch_size);
                    d_bias[k] = 0.0f;
                    for (int l = 0; l < 5; ++l) {
                        kernel[k][l] -= LEARNING_RATE * (d_kernel[k][l] / current_batch_size);
                        d_kernel[k][l] = 0.0f;
                    }
                }
            }
        }

        cout << "Epoch " << epoch + 1 << " -> Average Loss: " << (epoch_loss / 87554.0f) << endl;
    }

    // --- 4. EXPORTING ALL PARAMETERS FOR INFERENCE CODE (.emo FILE) ---
    cout << "\n-> Saving all updated network weights and parameters to model.emo..." << endl;
    ofstream outFile("model.emo", ios::binary);
    if (outFile.is_open()) {
        // [1] Layer 1 Weights & Biases
        for (int i = 0; i < 32; ++i) {
            outFile.write(reinterpret_cast<char*>(kernel[i]), 5 * sizeof(float));
        }
        outFile.write(reinterpret_cast<char*>(bias), 32 * sizeof(float));

        // [2] Layer 2 Weights & Biases
        for (int k = 0; k < OUT_CHANNELS_L2; ++k) {
            for (int l = 0; l < KERNEL_SIZE_L2; ++l) {
                outFile.write(reinterpret_cast<char*>(kernel_L2[k][l]), IN_CHANNELS_L2 * sizeof(float));
            }
        }
        outFile.write(reinterpret_cast<char*>(bias_L2), 64 * sizeof(float));

        // [3] Fully Connected Dense Weights & Biases
        for (int i = 0; i < NUM_CLASSES; ++i) {
            outFile.write(reinterpret_cast<char*>(fc_weights[i]), FLATTEN_SIZE * sizeof(float));
        }
        outFile.write(reinterpret_cast<char*>(fc_bias), NUM_CLASSES * sizeof(float));

        outFile.close();
        cout << "-> Export Successful! model.emo is optimized and ready for your inference engine." << endl;
    } else {
        cerr << "Error: Could not open file to write model parameters." << endl;
    }

    // Dynamic Memory Cleanup
    for (int i = 0; i < 32; ++i) {
        delete[] kernel[i];
        delete[] d_kernel[i];
    }
    delete[] kernel;
    delete[] d_kernel;

    for (int k = 0; k < OUT_CHANNELS_L2; ++k) {
        for (int l = 0; l < KERNEL_SIZE_L2; ++l) {
            delete[] kernel_L2[k][l];
            delete[] d_kernel_L2[k][l];
        }
        delete[] kernel_L2[k];
        delete[] d_kernel_L2[k];
    }
    delete[] kernel_L2;
    delete[] d_kernel_L2;

    for (int i = 0; i < NUM_CLASSES; ++i) {
        delete[] fc_weights[i];
        delete[] d_fc_weights[i];
    }
    delete[] fc_weights;
    delete[] d_fc_weights;

    return 0;
}