#include <iostream>
#include <cmath>
#include <vector>

// Compute the loss function
double ComputeLossFunction(const std::vector<double>& x, const std::vector<double>& y, double w)
{
    double loss = 0.0;
    for(size_t i = 0; i < x.size(); i++)
    {
        double pred = x[i] * w;
        loss += pow(pred - y[i], 2);
    }
    return loss / x.size();
}

int main()
{
    std::vector<double> x = {1,2,3,4};
    std::vector<double> y = {2,4,6,8};

    double w = 0.0;
    double learning_rate = 0.01;
    int epochs = 1000;


    double beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8;
    double m = 0, v = 0;

    for(int epoch = 0; epoch < epochs; epoch++)
    {
        double gradient = 0.0;
        for(size_t  i = 0; i < x.size(); i++)
        {
            gradient += (w * x[i] - y[i]) * x[i];
        }
        gradient /= x.size();
        m = beta1 * m + (1 - beta1) * gradient;
        v = beta2 * v + (1 - beta2) * gradient * gradient;

        double m_hat = m / (1 - std::pow(beta1, epoch + 1));
        double v_hat = v / (1 - std::pow(beta2, epoch + 1));

        w -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

        if(epoch % 100 == 0)
        {
            printf("\n Epoch: %d Loss: %.3f Weight: %.3f",epoch, ComputeLossFunction(x, y, w), w);
        }
    }

}