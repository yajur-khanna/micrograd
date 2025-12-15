# microrad

Micrograd — Clean Educational Reimplementation

A minimal, from-scratch automatic differentiation engine inspired by Andrej Karpathy’s micrograd, implemented with clarity and pedagogical intent.

This project demonstrates how reverse-mode autodiff (backpropagation) works under the hood by building a tiny scalar-based computation graph engine without relying on PyTorch, TensorFlow, or JAX.

## Features

- Scalar automatic differentiation

- Dynamic computation graph

- Reverse-mode backpropagation

- Supported operations:

  - +, -, *, -
  - ReLU, Sigmoid

- No external ML libraries

# Project Goals

This project is not intended to be:

- A performant deep learning framework

- A drop-in replacement for PyTorch/JAX

- Feature-complete beyond core autodiff mechanics

It is intended to:

- Make backpropagation mechanically obvious

- Help students understand how gradients flow

- Serve as a foundation for extending toward neural networks

- Act as a reference when learning larger frameworks

# Core Abstraction

At the heart of the system is a Value object that stores:

- data: the scalar value

- grad: the accumulated gradient

- _prev: parent nodes in the computation graph

- _op: the operation that produced the node

- _backward(): local gradient rule

Backpropagation is performed by:

- Topologically sorting the computation graph

- Traversing it in reverse

- Applying the chain rule at each node

## Example Usage

<code> from micrograd import Value

x = Value(2.0)
y = Value(-3.0)
z = x * y + x**2

z.backward()

print(z.data)   # forward pass result
print(x.grad)   # dz/dx
print(y.grad)   # dz/dy <\code>

This example builds a computation graph (given in the figure below) dynamically and computes gradients via reverse-mode autodiff.
