pub mod kernel;
mod error;
mod util;

use nalgebra::{DMatrix, RowDVector, ComplexField, RealField, Scalar, Field};
use num::Float;

use kernel::Kernel;

use error::PcaError;
use util::sort_indices_descending;


///
/// Define a Kernel PCA configuration
/// 
#[derive(Clone, Debug)]
pub struct KernelPca<T: Float> {
    /// The kernel function
    pub kernel: Kernel<T>,
    /// The embedding dimension
    pub embed_dim: usize
}

impl <T: Float + ComplexField + RealField> KernelPca<T> {

    ///
    /// Constructs a new KernelPCA instance
    /// 
    /// # Arguments
    /// 
    /// * `kernel` - The kernel function
    /// * `embed_dim` - The desired embedding dimension
    /// 
    pub fn new(kernel: Kernel<T>, embed_dim: usize) -> KernelPca<T> {
        KernelPca { kernel, embed_dim }
    }

    ///
    /// Applies Kernel PCA to the provided input data, outputting
    /// the projected embeddings
    /// 
    /// # Arguments
    /// 
    /// * `data` - The input data, as a vector of feature vectors
    /// 
    pub fn apply(&self, data: &Vec<Vec<T>>) -> Result<Vec<Vec<T>>, PcaError> {
        self.validate(data)?;
        // The linear kernel is equivalent to vanilla PCA, so let's
        // bypass the construction of the kernel matrix and just use SVD
        if let Kernel::Linear = self.kernel {
            let x = center_data(data)?;
            let svd = x.svd(true, false);
            let sigma = DMatrix::from_diagonal(&svd.singular_values.rows(0, self.embed_dim));
            let embeddings = svd
            .u
            .ok_or(PcaError::computation_failure("SVD Failure"))?
            .columns(0, self.embed_dim) * sigma;
            return Ok(
                embeddings
                .row_iter()
                .map(|row| row.iter().map(|&val| val)
                .collect()
            ).collect())
        }
        // Otherwise construct the (centered) kernel matrix and use eigen decomposition
        let k = self.form_kernel_matrix(data);
        let k_center = center_kernel_matrix(&k)?;
        let factorization = k_center.symmetric_eigen();
        let indices = sort_indices_descending(factorization.eigenvalues.as_slice());
        let mut embeddings = vec![vec![T::zero(); self.embed_dim]; data.len()];
        for j in 0..self.embed_dim {
            let ind = indices[j];
            let eigenvector = factorization.eigenvectors.column(ind);
            let eigenvalue_sqrt = Float::sqrt(factorization.eigenvalues[ind]);
            for i in 0..data.len() {
                embeddings[i][j] = eigenvector[i] * eigenvalue_sqrt;
            }
        }
        return Ok(embeddings);
    }

    fn form_kernel_matrix(&self, x: &Vec<Vec<T>>) -> DMatrix<T> {
        let n = x.len();
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..i {
                let kij = self.kernel.compute(&x[i], &x[j]);
                k[(i, j)] = kij;
                k[(j, i)] = kij;
            }
            k[(i, i)] = self.kernel.compute(&x[i], &x[i]);
        }
        return k;
    }

    fn validate(&self, data: &Vec<Vec<T>>) -> Result<(), PcaError> {
        if data.len() == 0 {
            return Err(PcaError::invalid_data("Input data has no records"));
        }
        let dim = data[0].len();
        if dim == 0 {
            return Err(PcaError::invalid_data("Input data has a dimensionality of zero"));
        }
        for row in data {
            if row.len() != dim {
                return Err(PcaError::invalid_data("Input data has inconsistent dimensionality across records"));
            }
        }
        if self.embed_dim == 0 {
            return Err(PcaError::invalid_config("Embedding dimension must be positive"));
        }
        if self.embed_dim > dim {
            return Err(PcaError::invalid_config("Embeddind dimension must be <= data dimension"));
        }
        Ok(())
    }
}

// Center the kernel matrix for kernel PCA
fn center_kernel_matrix<T: Float + Scalar + Field>(k: &DMatrix<T>) -> Result<DMatrix<T>, PcaError> {
    let dim = k.nrows();
    let tdim = T::from(dim).ok_or(PcaError::computation_failure("Unable to convert dimension to float"))?;
    let q = DMatrix::from_element(dim, dim, T::one() / tdim);
    let r = DMatrix::identity(dim, dim) - q;
    return Ok((&r * k) * &r);
}

// Center the input data for standard PCA
fn center_data<T: Float + Scalar + Field>(x: &Vec<Vec<T>>) -> Result<DMatrix<T>, PcaError> {
    let dim = x[0].len();
    let n = T::from(x.len()).ok_or(PcaError::computation_failure("Unable to convert data length to float"))?;
    let mut means = vec![T::zero(); dim];
    for row in x {
        for (j, &val) in row.iter().enumerate() {
            means[j] += val;
        }
    }
    for j in 0..dim {
        means[j] /= n;
    }
    Ok(DMatrix::from_rows(&x.iter().map(|row| {
        RowDVector::from_iterator(
            dim, 
            row
            .iter()
            .zip(means.iter())
            .map(|(&v, &m)| v - m)
        )
    }).collect::<Vec<_>>()))
}
