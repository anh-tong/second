
At the very beginning step of learning Pytorch, one may work some basic steps to enable auto-differentiation in this framework.
1. Create an optimizer object ```optimizer``` which can be SGD, Adam, RMSprop, etc.
2. Compute loss ```loss``` from variables or module.
3. Perform ```loss.backward()```.

 What happens here is that Pytorch will automatically compute the gradients of all the variables associated with or participated in computing ```loss```. Let's say one of the variable ```p```. As we run ```backward```, ```p``` will be added a new attribute ```p.grad``` containing the gradient information with respect to this variable.

 Now, when performing ```optimizer.step()```, the optimizer will look at the current value of the variable ```p``` and the gradient ```p.grad``` to perform gradient descent steps. 

 So, that was what is inside these compact lines of code. How do we customize this when we need to make our own gradient information? 

In fact, the ```p.grad``` is simply taken from the result after calling a function ```torch.autograd.grad```. This function is important to understand so we can utilize it effectively. From Pytorch documentation, this auto-differentation function looks like
```
torch.autograd.grad(outputs,
                    inputs,
                    grad_outputs,
                    create_graph=False,
                    retain_graph=False,
                    allow_used=False)
```
There are some cases we may consider,
1. Simply compute gradients of a loss variable ```loss``` with respect to a Pytorch module parameters (neural network parameters) as ```model.parameters()``` which should be passed to the function as ```list(model.parameters())``` or ```tuple(model.parameters())```. So our gradient will be
```
gradient = torch.autograd.grad(loss,
                    tuple(model.parameters())
```
2. Jacobian vector product. We use the ```grad_outputs``` option to indicate the vector of the product with Jacobian. Note that if our function maps from R^m to R^n, the size of Jacobi is m by n, and therefore our vector dimesion should be n. Or in the other hand, ```outputs``` and ```grad_outputs``` should have the same dimension

```
torch.autograd.grad(outputs,
                    inputs,
                    grad_outputs)
```
3. Higher order differentiation. This can be done by turn on the option ```create_graph=True```, so the result can further put into another ```torch.autograd.grad```.
