import torch
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector


def jvp(outputs, inputs, vector, create_graph=False):
    """Jacobian vector product
        This version where vector is flatten
    """

    if isinstance(outputs, tuple):
        dummy = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    else:
        dummy = torch.zeros_like(outputs, requires_grad=True)

    jacobian = grad(outputs,inputs, grad_outputs=dummy, create_graph=True)
    Jv = grad(parameters_to_vector(jacobian), dummy, grad_outputs=vector, create_graph=create_graph)
    return parameters_to_vector(Jv)

def hvp(loss, parameters, vector):
    """
        Hessian vector product - Flatten version
    """
    gradient = grad(loss, parameters, create_graph=True)
    return jvp(gradient, parameters, vector)

def hvp_v2(loss, parameters, vector):

    gradients = grad(loss, parameters, create_graph=True)
    Hv = grad(gradients, parameters, vector)
    return Hv