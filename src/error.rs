
#[derive(Clone, Debug)]
pub enum PcaError {
    ComputationFailure(String),
    InvalidConfig(String),
    InvalidData(String)
}

impl PcaError {

    pub fn computation_failure(message: impl Into<String>) -> PcaError {
        PcaError::ComputationFailure(message.into())
    }

    pub fn invalid_config(message: impl Into<String>) -> PcaError {
        PcaError::InvalidConfig(message.into())
    }

    pub fn invalid_data(message: impl Into<String>) -> PcaError {
        PcaError::InvalidData(message.into())
    }
}
