#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Activation function: Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Function to compute Mean Squared Error (MSE)
double compute_mse_loss(const vector<double>& y_pred, const vector<double>& y_true) {
    double loss = 0.0;
    for (size_t i = 0; i < y_pred.size(); i++) {
        loss += pow(y_pred[i] - y_true[i], 2);
    }
    return loss / y_pred.size();
}

// Function to compute Mean Absolute Error (MAE)
double compute_mae_loss(const vector<double>& y_pred, const vector<double>& y_true) {
    double loss = 0.0;
    for (size_t i = 0; i < y_pred.size(); i++) {
        loss += abs(y_pred[i] - y_true[i]);
    }
    return loss / y_pred.size();
}

int main() {
    srand(time(0)); // Seed for random number generation

    // Training data (y = 2x)
    vector<double> x = {1, 2, 3};
    vector<double> y = {2, 4, 6};
    
    // Initialize weights randomly for input-to-hidden and hidden-to-output layers
    double w_input_hidden = (double)rand() / RAND_MAX;  // Random weight for input to hidden
    double w_hidden_output = (double)rand() / RAND_MAX; // Random weight for hidden to output
    double learning_rate = 0.0009;
    int epochs = 1000;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        vector<double> y_pred;
        double gradient_hidden = 0.0;
        double gradient_output = 0.0;
        
        for (size_t i = 0; i < x.size(); i++) {
            // Forward pass
            double hidden_output = sigmoid(x[i] * w_input_hidden);
            double predicted = hidden_output * w_hidden_output;
            y_pred.push_back(predicted);
            
            // Compute gradients using MSE loss
            double error = predicted - y[i];
            gradient_output += error * hidden_output; // Gradient for output layer
            gradient_hidden += error * w_hidden_output * sigmoid_derivative(hidden_output) * x[i]; // Gradient for hidden layer
        }
        
        gradient_output /= x.size();
        gradient_hidden /= x.size();
        
        // Update weights
        w_hidden_output -= learning_rate * gradient_output;
        w_input_hidden -= learning_rate * gradient_hidden;
        
        // Compute losses
        double mse_loss = compute_mse_loss(y_pred, y);
        double mae_loss = compute_mae_loss(y_pred, y);
        
        // Print progress every 100 epochs
        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << " MSE Loss: " << mse_loss << " MAE Loss: " << mae_loss;
            cout << " w_input_hidden: " << w_input_hidden << " w_hidden_output: " << w_hidden_output << endl;
        }
    }

    cout << "Final Weights - Input to Hidden: " << w_input_hidden << " | Hidden to Output: " << w_hidden_output << endl;
    return 0;
}
