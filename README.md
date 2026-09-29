# Deep Learning Mathematics in MATLAB

Derive the gradients of a fully connected neural network, then follow them through two MATLAB implementations. Both scripts train a house-price regressor with ordinary matrix operations and manually computed derivatives; no automatic differentiation is used.

- [Gradient.m](Gradient.m) writes the chain rule as expanded expressions and recomputes intermediate activations.
- [Backpropagation.m](Backpropagation.m) stores forward activations and reuses error signals while working backward through the layers.

Start with the [batch notation](#batch-notation), follow the [forward pass](#forward-pass-and-loss), then compare the [backward pass](#backward-pass) with the two scripts.

## Run the examples

Use **MATLAB R2019a or later**: the scripts call [`readmatrix`](https://www.mathworks.com/help/matlab/ref/readmatrix.html), which was introduced in R2019a. The source uses base MATLAB functions; it does not call Deep Learning Toolbox APIs.

Clone the repository, open MATLAB, and set the current folder to the repository root:

```sh
git clone https://github.com/kapshaul/deep-learning-math.git
cd deep-learning-math
```

Run either script in the MATLAB Command Window:

```matlab
Gradient
```

```matlab
Backpropagation
```

Both read relative paths under `Dataset/`, clear the workspace at startup, print `Validation Loss` every 10 epochs, and print elapsed time when training finishes. They leave the final weights and biases in the workspace; they do not save a model or create plots. Running the second script clears the first script's variables.

## Repository contents

```text
.
├── Gradient.m                    # Expanded gradient expressions
├── Backpropagation.m             # Forward cache and backward error signals
├── README.md
├── LICENSE
└── Dataset/
    ├── train_house.csv           # Used for training by both scripts
    ├── test_house.csv            # Used for the printed validation loss
    ├── train_accident.csv
    ├── valid_accident.csv
    ├── test_accident.csv
    ├── train_smoke.csv
    ├── valid_smoke.csv
    ├── test_smoke.csv
    ├── MNIST_train_image.mat
    ├── MNIST_train_label.mat
    ├── MNIST_test_image.mat
    ├── MNIST_test_label.mat
    └── Extra/
        ├── train_insurance.csv
        ├── valid_insurance.csv
        └── test_insurance.csv
```

Only the two house CSVs are loaded by the current examples. The accident, smoke, insurance, and MNIST files are additional data; using them requires changing the import, feature/target selection, and model dimensions as appropriate.

## Data and training defaults

The house CSVs each have a header and 38 columns. `train_house.csv` contains 10,000 data rows; `test_house.csv` contains 5,597. The scripts use positional indexing:

```matlab
X = data_train(:, 1:37);  % Input features, including the column named dummy
y = data_train(:, 38);   % Target: price
```

The inputs include date fields, housing attributes, location, and encoded categories. `readmatrix` automatically detects import settings, including header lines; the scripts perform no further feature scaling, missing-value handling, or target transformation. The CSV column order therefore matters.

| Setting | Value in both scripts |
| --- | --- |
| Architecture | 37 inputs → 128 hidden units → 64 hidden units → 1 output |
| Hidden activation | ReLU |
| Output activation | Identity: a scalar regression prediction |
| Objective | Squared prediction error, with the averaging detail explained below |
| Optimizer | Manual mini-batch gradient descent, without momentum |
| Learning rate | `1e-4` |
| Epochs | `100` |
| Configured batch size | `32` |
| Initialization | `randn` weights and zero biases |

The sigmoid and tanh functions defined in each script are unused; the forward pass and gradients use ReLU.

## Batch notation

The full training array `X` in the scripts stores one example per **row**. Inside the training loop, `x = X(i:batch_end, :)'` transposes a batch so each example becomes a **column**. The equations below use that column convention throughout.

Let $B$ be the actual number of examples in the current batch, and let $\mathbf{1}_B$ be a column of $B$ ones.

| Symbol | MATLAB variable | Shape |
| --- | --- | --- |
| $X_b$ | `x` | $37 \times B$ |
| $Y_b$ | `y_true` | $1 \times B$ |
| $W_1, b_1$ | `W1`, `b1` | $128 \times 37$, $128 \times 1$ |
| $W_2, b_2$ | `W2`, `b2` | $64 \times 128$, $64 \times 1$ |
| $W_3, b_3$ | `W3`, `b3` | $1 \times 64$, $1 \times 1$ |
| $Z_1, A_1$ | `z1`, `a1` | $128 \times B$ |
| $Z_2, A_2$ | `z2`, `a2` | $64 \times B$ |
| $\widehat{Y}, E$ | `y_pred`, `error` | $1 \times B$ |

Matrix multiplication uses `*`; the Hadamard product $\odot$ uses `.*` and multiplies corresponding entries. See MATLAB's [array versus matrix operations](https://www.mathworks.com/help/matlab/matlab_prog/array-vs-matrix-operations.html). All quantities here are real, so the scripts' transpose operator `'` gives the transposes used below.

## Forward pass and loss

With $\sigma(z)=\max(0,z)$ applied element by element:

```math
\begin{aligned}
Z_1 &= W_1 X_b + b_1\mathbf{1}_B^T, & A_1 &= \sigma(Z_1), \\
Z_2 &= W_2 A_1 + b_2\mathbf{1}_B^T, & A_2 &= \sigma(Z_2), \\
\widehat{Y} &= W_3 A_2 + b_3\mathbf{1}_B^T, & E &= \widehat{Y}-Y_b.
\end{aligned}
```

Each bias is shared by all examples in a batch. MATLAB implements this broadcast directly with expressions such as `W1 * x + b1`; see [compatible array sizes and implicit expansion](https://www.mathworks.com/help/matlab/matlab_prog/compatible-array-sizes-for-basic-operations.html).

For one scalar output per example, the batch mean squared error is

```math
J = \frac{1}{B}\sum_{j=1}^{B} E_{1j}^{\,2},
\qquad
\frac{\partial J}{\partial\widehat{Y}} = \frac{2}{B}E.
```

The following derivation uses the actual batch size $B$. Both scripts instead divide training gradients by the configured `batch_size`; the final partial batch is discussed under [implementation details](#implementation-details-to-keep-in-mind).

## Backward pass

Define $\Delta_\ell=\partial J/\partial Z_\ell$, with $Z_3=\widehat{Y}$. ReLU's derivative is 1 for positive inputs and 0 for negative inputs; at zero, these scripts choose 0 via `ReLU_deriv = @(x) x > 0`.

Start at the linear output, then apply the chain rule through each hidden layer:

```math
\begin{aligned}
\Delta_3 &= \frac{2}{B}E, \\
\Delta_2 &= (W_3^T\Delta_3)\odot\sigma'(Z_2), \\
\Delta_1 &= (W_2^T\Delta_2)\odot\sigma'(Z_1).
\end{aligned}
```

The matrix product propagates error to the previous layer; the element-wise product applies that layer's activation derivative. The shapes are $1\times B$, $64\times B$, and $128\times B$, respectively.

For each affine layer, multiply its error signal by the transpose of its input to obtain the weight gradient. Sum across examples to obtain the gradient of its shared bias:

```math
\begin{aligned}
\frac{\partial J}{\partial W_3} &= \Delta_3 A_2^T,
& \frac{\partial J}{\partial b_3} &= \Delta_3\mathbf{1}_B, \\
\frac{\partial J}{\partial W_2} &= \Delta_2 A_1^T,
& \frac{\partial J}{\partial b_2} &= \Delta_2\mathbf{1}_B, \\
\frac{\partial J}{\partial W_1} &= \Delta_1 X_b^T,
& \frac{\partial J}{\partial b_1} &= \Delta_1\mathbf{1}_B.
\end{aligned}
```

Each gradient has the same shape as its parameter. For example, $(64\times B)(B\times128)$ gives the $64\times128$ gradient for $W_2$. In MATLAB, `sum(delta2, 2)` sums the batch columns and returns a $64\times1$ bias gradient. This matches the documented behavior of [`sum(A,2)`](https://www.mathworks.com/help/matlab/ref/double.sum.html).

Finally, update every parameter using the same forward pass and its gradients:

```math
W_\ell \leftarrow W_\ell-\eta\frac{\partial J}{\partial W_\ell},
\qquad
b_\ell \leftarrow b_\ell-\eta\frac{\partial J}{\partial b_\ell},
\qquad \ell\in\{1,2,3\}.
```

Both scripts compute all six gradients before changing any weights or biases.

## Expanded gradients and cached backpropagation

Substituting the error signals into the weight gradients gives the expanded chain rule:

```math
\begin{aligned}
\frac{\partial J}{\partial W_3}
&= \frac{2}{B}EA_2^T, \\
\frac{\partial J}{\partial W_2}
&= \frac{2}{B}\big[(W_3^TE)\odot\sigma'(Z_2)\big]A_1^T, \\
\frac{\partial J}{\partial W_1}
&= \frac{2}{B}\Big[\big(W_2^T[(W_3^TE)\odot\sigma'(Z_2)]\big)
\odot\sigma'(Z_1)\Big]X_b^T.
\end{aligned}
```

These are the expressions in `Gradient.m`, with $A_1$, $A_2$, $Z_1$, and $Z_2$ replaced by their forward expressions. Its bias gradients use the same error factors, summed across the batch.

`Backpropagation.m` names those repeated quantities `z1`, `a1`, `z2`, `a2`, and `delta1`–`delta3`. Reusing them avoids recomputing the same activations and makes the chain rule easier to inspect. For identical parameters and the same batch, both forms express the same gradients, up to floating-point evaluation order. Their complete training runs also depend on initialization and sample order.

## Implementation details to keep in mind

- **Final batch scaling.** There are 10,000 training rows: 312 full batches of 32 and a final batch of 16. Both scripts still use `2/batch_size`, so the final batch's gradients are half the gradients of its true mean squared error. The equations above use $2/B$; reproducing the current implementation requires $2/32$ on every batch.
- **Reported metric.** The helper `loss = @(error) sum((error).^2)` returns a sum of squared errors for the row-vector predictions. The evaluation code divides it by the number of evaluation rows, producing MSE.
- **Split naming.** `test_house.csv` is assigned to `data_valid` and evaluated every 10 epochs under the label `Validation Loss`. There is no separate validation file or final untouched test evaluation in these scripts. If you tune settings using this output, treat that file as validation data.
- **Different sample orders.** `Gradient.m` shuffles the training rows after each epoch; `Backpropagation.m` keeps their original order. Neither sets an RNG seed. Set `rng` explicitly before each run if you need repeatable initialization, and account for the different ordering when comparing results.
- **Numerical behavior.** The scripts use unscaled `randn` initialization and apply no additional data scaling, gradient clipping, or finite-value checks. They are derivation examples, not a documented convergence benchmark. Check the printed losses for non-finite values before interpreting a run.
- **Adapting the examples.** Changing datasets or output dimensions also requires revisiting feature/target slices, activation and loss choices, and reductions over examples and outputs. The derivation here assumes one scalar regression target.

## Related repositories

- [deep-learning](https://github.com/kapshaul/deep-learning): sine-function approximation and XOR classification.
- [deep-learning-cnn](https://github.com/kapshaul/deep-learning-cnn): convolutional networks and MNIST classification.
- [deep-learning-rnn](https://github.com/kapshaul/deep-learning-rnn): recurrent networks, LSTMs, and sequential MNIST.

## License

This repository is distributed under the [MIT License](LICENSE).
