#include<iostream>
#include<random>
using namespace std;

void randomgen(int row, int column, float** output, float min_val = -1.0f, float max_val = 1.0f) {
    
    mt19937 gen(random_device{}());
    uniform_real_distribution<float> dis(min_val, max_val);

    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            output[i][j] = dis(gen);
        }
    }
}

int main() {
    int row = 3;
    int column = 4;

    // ১. প্রথমে মেমরি অ্যালোকেট করো (তোমার অ্যাপ্লিকেশনের অ্যারে)
    float** my_array = new float*[row];
    for (int i = 0; i < row; ++i) {
        my_array[i] = new float[column];
    }

    // ২. ফাংশন কল করো এবং আউটপুট অ্যারে পাস করো
    randomgen(row, column, my_array, -0.5f, 0.5f);

    // ৩. আউটপুট চেক করার জন্য প্রিন্ট (ঐচ্ছিক)
    std::cout << "Array successfully filled with random floats:\n";
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            std::cout << my_array[i][j] << "\t";
        }
        std::cout << "\n";
    }
}