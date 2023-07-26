# kernel-pca

A Rust implementation of Kernel PCA

## Example

```rust
use kernel_pca::*;

fn main() {

    let data = vec![
        vec![0.35, 0.32, 0.74, 0.33, 0.03],
        vec![0.21, 0.53, 0.39, 0.17, 0.09],
        vec![0.75, 0.09, 0.12, 0.09, 0.45],
        vec![0.94, 0.78, 0.02, 0.63, 0.71],
        vec![0.88, 0.45, 0.83, 0.79, 0.51],
        vec![0.57, 0.92, 0.26, 0.22, 0.36],
        vec![0.38, 0.08, 0.28, 0.25, 0.74],
        vec![0.62, 0.05, 0.90, 0.28, 0.62],
        vec![0.15, 0.80, 0.98, 0.77, 0.89],
        vec![0.62, 0.34, 0.67, 0.85, 0.14],
    ];

    let kernel = Kernel::SquaredExponential { gamma: 0.5 };
    let embed_dim = 2;
    let kpca = KernelPca::new(kernel, embed_dim);
    let embeddings = kpca.apply(&data).unwrap();

    for row in embeddings {
        println!("{:?}", row)
    }
}
```
```
[0.03371512105021725, -0.36025153375128244]
[-0.2281607317323001, -0.19155514840137025]
[-0.44213488951130936, -0.0799834690614372]
[-0.16665423805428528, 0.6013248234227786]
[0.3798813322081997, 0.09531917376191743]
[-0.23894754961573755, 0.2872573358397481]
[-0.2633418871787093, -0.13648271826403918]
[0.12763621239352643, -0.30345383370697243]
[0.5241580180127576, 0.17761656446265375]
[0.2738486124276399, -0.08979119430199604]
```

## Available Kernels

### Squared Exponential

The Squared Exponential (RBF) kernel takes the form:

k(x, x') = e<sup>-&gamma; (x - x')&sup2;</sup>

### Rational Quadratic

The Rational Quadratic kernel takes the form:

k(x, x') = (1 + (&gamma;/&alpha;) (x - x')&sup2;)<sup>-&alpha;</sup>

### Linear

The Linear kernel takes the form:

k(x, x') = x * x'

NOTE: This kernel is mathematically equivalent to standard PCA. As such, this library defaults to a standard PCA implementation when the Linear kernel is selected, which avoids computing the kernel matrix.
