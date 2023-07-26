pub mod kernel;
mod error;

use nalgebra::{DMatrix, RowDVector, ComplexField, RealField, Scalar, Field};
use num::Float;

pub use kernel::Kernel;
pub use error::PcaError;


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
        // For the linear kernel, we just use vanilla PCA and avoid the kernel matrix
        let x = match self.kernel {
            Kernel::Linear => center_data(data)?,
            _ => center_kernel_matrix(&self.form_kernel_matrix(data))?
        };
        let svd = x.svd(true, false);
        let sv_selection = svd.singular_values.rows(0, self.embed_dim);
        // Remember we don't need to take the square root for the linear case
        let sigma = match self.kernel {
            Kernel::Linear => DMatrix::from_diagonal(&sv_selection),
            _ => DMatrix::from_diagonal(&sv_selection.map(|v| Float::sqrt(v)))
        };
        let u = svd
        .u
        .ok_or(PcaError::computation_failure("SVD Failure"))?;
        let signs = determine_signs(&u, self.embed_dim);
        let u_selection = u.columns(0, self.embed_dim);
        let embeddings = u_selection * sigma;
        return Ok(
            embeddings
            .row_iter()
            .map(|row| {
                row.iter()
                .enumerate()
                .map(|(j, &val)| val * signs[j])
                .collect()
            })
            .collect()
        )
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

fn determine_signs<T: Float>(u: &DMatrix<T>, dim: usize) -> Vec<T> {
    u.columns(0, dim).column_iter().map(|column| {
        let mut max_abs_elem = T::zero();
        let mut flip = false;
        for &val in column.iter() {
            if val.abs() > max_abs_elem {
                max_abs_elem = val.abs();
                if val < T::zero() {
                    flip = true;
                } else {
                    flip = false;
                }
            }
        }
        if flip {
            T::one().neg()
        } else {
            T::one()
        }
    }).collect()
}
