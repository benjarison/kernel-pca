use std::fmt;


///
/// Defines various errors
/// 
#[derive(Clone, Debug)]
pub enum PcaError {
    /// Indicates a failure encountered during Kernel PCA computation
    ComputationFailure(String),
    /// Indicates an invalid Kernel PCA configuration
    InvalidConfig(String),
    /// Indicates invalid input data
    InvalidData(String)
}

impl PcaError {

    ///
    /// Constructs a new ComputationFailure instance
    /// 
    /// # Arguments
    /// 
    /// * `message` - The error message
    /// 
    pub fn computation_failure(message: impl Into<String>) -> PcaError {
        PcaError::ComputationFailure(message.into())
    }

    ///
    /// Constructs a new InvalidConfig instance
    /// 
    /// # Arguments
    /// 
    /// * `message` - The error message
    /// 
    pub fn invalid_config(message: impl Into<String>) -> PcaError {
        PcaError::InvalidConfig(message.into())
    }

    ///
    /// Constructs a new InvalidData instance
    /// 
    /// # Arguments
    /// 
    /// * `message` - The error message
    /// 
    pub fn invalid_data(message: impl Into<String>) -> PcaError {
        PcaError::InvalidData(message.into())
    }
}

impl fmt::Display for PcaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::ComputationFailure(message) => message,
            Self::InvalidConfig(message) => message,
            Self::InvalidData(message) => message
        };
        write!(f, "{}", message)
    }
}
