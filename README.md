# Notes of Key Concepts

### Perceptron:

> Simplest form of a neuron  
> Takes in a weighted (w) sum (x) + a bias (b) as a input and returns output Z **(linear function)**
>
> #### Simply put, multiple perceptron linked together form a Neural Network

### Activation function

> Gives you the probability of an output z from linear function being correct

### Sigmoid Function (activation function)

> The sigmoid function gives the probability that the input belongs to the positive class often represented as 1.  
> It maps any input z to a value between 0 and 1, interpreting it as the probability that the output is 1.  
> The closer we are to 0.5, the model is uncertain of which class (0 or 1) the input z belongs to

## So a Neuron is an ouput Z and a probability of Z being correct

### Cost Function

> A cost function is the difference between the model's output when it has real data as a input vs the actual output of the real data  
> Multiply the probability of each output being correct (activation function) together **->** converges to 0 therefore we use logarithmic function  
> Log of all these will not give the same result, but it will keep the same order as the logarithmic function is **monotonically increasing**.
> Log Loss function

### Gradient Descent

> MLE (Maximal Likelihood Estimation) is not a single function.  
> We need to take our linear function and adjust our **W** parameters and our bias (b) until we minimise our errors (until it is the most likely)(Log Loss)
> We need to determine how adjusting our W parameters affect our likelihood which is why we need the **Gradient**  
> **The Gradient is the derivative of the Cost function**  
> The function must be convex, meaning that we dont have local minimums

## Important definiton of functions I will be using

![Alt text](./functions.png)
