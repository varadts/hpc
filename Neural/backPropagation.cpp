#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Function to compute Mean Squared Error (MSE)
double compute_loss(const vector<double>& x, const vector<double>& y, double w) {
    double loss = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        double pred = w * x[i];
        loss += pow(pred - y[i], 2);
    }
    return loss / x.size();
}

int main() {
    // Training data (y = 2x)
    vector<double> x = {1, 2, 3, 4, 5};
    vector<double> y = {2, 4, 6, 8, 10};
    
    double w = 0.0;  // Initial weight
    double learning_rate = 0.01;
    int epochs = 1000; // Number of training iterations

    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        double gradient = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            gradient += (w * x[i] - y[i]) * x[i]; // Derivative of MSE
        }
        gradient /= x.size();
        w -= learning_rate * gradient; // Update weight using gradient descent
        
        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << " Loss: " << compute_loss(x, y, w) << " Weight: " << w << endl;
        }
    }

    cout << "Final weight: " << w << endl;
    return 0;
}
