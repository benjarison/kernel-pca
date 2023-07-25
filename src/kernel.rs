use num::Float;

///
/// Defines various kernel functions
/// 
#[derive(Clone, Debug)]
pub enum Kernel<T: Float> {
    /// Linear kernel of the form x * x' (Note that this is equivalent to standard PCA)
    Linear,
    /// Rational Quadratic kernel of the form (1 + gamma * (x - x')^2)^(-alpha)
    RationalQuadratic { gamma: T, alpha: T },
    /// Squared Exponential (or RBF) kernel of the form exp(-gamma * (x - x')^2)
    SquaredExponential { gamma: T }
}

impl <T: Float> Kernel<T> {

    ///
    /// Construct a new Linear kernel
    /// 
    pub fn linear() -> Kernel<T> {
        Kernel::Linear
    }

    ///
    /// Construct a new Rational Quadratic kernel
    /// 
    /// # Arguments
    /// 
    /// * `gamma` - The gamma scale value
    /// * `alpha` - The alpha exponent value
    /// 
    pub fn rational_quadratic(gamma: T, alpha: T) -> Kernel<T> {
        Kernel::RationalQuadratic { gamma, alpha }
    }

    ///
    /// Construct a new Squared Exponential kernel
    /// 
    /// # Arguments
    /// 
    /// * `gamma` - The gamma scale value
    /// 
    pub fn squared_exponential(gamma: T) -> Kernel<T> {
        Kernel::SquaredExponential { gamma }
    }

    ///
    /// Computes the kernel function for the provided points
    /// 
    /// # Arguments
    /// 
    /// * `a` - The first point
    /// * `b` - The second point
    /// 
    pub fn compute(&self, a: &[T], b: &[T]) -> T {
        match self {
            Self::Linear => compute_linear(a, b),
            Self::RationalQuadratic { gamma, alpha } => compute_rational_quadratic(a, b, *gamma, *alpha),
            Self::SquaredExponential { gamma } => compute_squared_exponential(a, b, *gamma)
        }
    }
}

// Specialized linear computation
// Note that this should never actually be used internally by this library
// Instead, we should use vanilla PCA and avoid constructing the kernel matrix
fn compute_linear<T: Float>(a: &[T], b: &[T]) -> T {
    a
    .iter()
    .zip(b.iter())
    .fold(T::zero(), |sum, (&a, &b)| sum + a * b)
}

// Specialized rational quadratic computation
fn compute_rational_quadratic<T: Float>(a: &[T], b: &[T], gamma: T, alpha: T) -> T {
    let ssd = a
    .iter()
    .zip(b.iter())
    .fold(T::zero(), |sum, (&a, &b)| {
        let diff = a - b;
        sum + diff * diff
    });
    return (T::one() + gamma * ssd).powf(-alpha);
}

// Specialized squared exponential computation
fn compute_squared_exponential<T: Float>(a: &[T], b: &[T], gamma: T) -> T {
    let ssd = a
    .iter()
    .zip(b.iter())
    .fold(T::zero(), |sum, (&a, &b)| {
        let diff = a - b;
        sum + diff * diff
    });
    return (-gamma * ssd).exp()
}
