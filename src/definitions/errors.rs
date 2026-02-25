use std::{error::Error, fmt};

#[derive(Debug)]
pub struct ValidationError {
    details: String,
}

impl Error for ValidationError {}

impl ValidationError {
    pub fn new<T: Into<String>>(details: T) -> Self {
        Self {
            details: details.into(),
        }
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Validation error: {}", self.details)
    }
}
