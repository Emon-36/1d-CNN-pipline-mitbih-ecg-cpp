
## Custom 1D-CNN Engine from Scratch in C++

An ultra-optimized, native C++ implementation of a 1D Convolutional Neural Network (1D-CNN) designed from the ground up for processing time-series ECG signals (MIT-BIH Dataset). This project bypasses high-level frameworks like TensorFlow or PyTorch, handling raw memory management, custom forward/backward propagation, and high-performance determinism natively.

## 🚀 Why This Project Exists?
Modern Deep Learning relies heavily on bloated frameworks that abstract away hardware realities. This engine was built to demonstrate **Systems Engineering for Edge AI**, implementing precise tensor tracking, backpropagation calculus, and a custom lightweight binary serialization format (`.emo`) to maintain zero memory fragmentation and sub-millisecond inference capability on resource-constrained hardware.

---

## 🛠️ Key Architectural Highlights & Engineering Feats

* **Zero Framework Reliance:** Every layer—from 1D Convolution, Max Pooling 1D, to Softmax and Cross-Entropy Loss—is coded using vanilla C++ structures.
* **Granular Backpropagation Control:** Custom-engineered backpropagation loops that meticulously track relative max-pooling indices to route gradients accurately during the backward pass.
* **Deterministic Weight Serialization:** Implements a custom structural binary file format (`.emo`) that dumps and loads layer weights/biases sequentially to maximize cache-locality and optimize memory alignment during edge inference.
* **Memory Optimization:** Designed specifically to target high-throughput environments by isolating training configurations from deployment binaries, eliminating OOM vulnerabilities on target hardware.

---

## 📐 Network Architecture & Hyperparameters

The model is structured for high-fidelity classification of single-channel electrocardiogram (ECG) signals into 5 distinct arrhythmia classes.

| Layer | Type | Specifications / Hyperparameters | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input** | Raw Signal | Time-series ECG sequence | `[Batch, 187, 1]` |
| **Layer 1** | Conv1D + ReLU | 32 Filters, Kernel Size = 5, Stride = 1 | `[Batch, 183, 32]` |
| **Pool 1** | Max Pooling 1D | Pool Size = 2, Stride = 2 | `[Batch, 91, 32]` |
| **Layer 2** | Conv1D + ReLU | 64 Filters, Kernel Size = 5, Stride = 1 | `[Batch, 87, 64]` |
| **Pool 2** | Max Pooling 1D | Pool Size = 2, Stride = 2 | `[Batch, 43, 64]` |
| **Flatten** | Reshape | Matrix Vectorization | `[Batch, 2752]` |
| **Output** | Dense + Softmax| 5 Neurons, Cross-Entropy Loss | `[Batch, 5]` |

### 🎛️ Training Configuration
* **Batch Size:** 100
* **Learning Rate ($\eta$):** 0.05
* **Dataset Size:** 87,554 Samples (MIT-BIH Arrhythmia Database)
* **Optimization:** Mini-batch Gradient Descent with synchronized batch-step update logic.

---

## 📂 File Serialization (`.emo` Specs)

To achieve lightning-fast loading states in production, the `.emo` file structure strictly maps variables as raw byte arrays:

1. **Conv1 Layers:** `kernel[32][5]` $\rightarrow$ `bias[32]` *(Floats)*
2. **Conv2 Layers:** `kernel_L2[64][5][32]` $\rightarrow$ `bias_L2[64]` *(Floats)*
3. **Dense Layers:** `fc_weights[5][2752]` $\rightarrow$ `fc_bias[5]` *(Floats)*

This clean layout enables any decoupled standard C++ inference script to read the weights directly into stack/static heap memory arrays via an `ifstream` binary stream in $O(1)$ allocation overhead.

---

## 💻 Technical Proficiencies Demonstrated

* **Mathematical Deep Learning:** Manual computation of partial derivatives for Softmax integrated with Categorical Cross-Entropy, and ReLu derivative routing.
* **Systems Programming:** Matrix handling, dynamic multi-dimensional array allocation/cleanup (`delete[]`), and minimizing memory footprint.
* **Edge Intelligence:** Designing pipelines focused on strict execution determinism and hardware-level deployment patterns.
