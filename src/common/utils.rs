use crate::definitions::errors::ValidationError;
use ulid::Ulid;

pub fn parse_ulid(id: &str) -> Result<Ulid, ValidationError> {
    match id.parse::<Ulid>() {
        Ok(u) => Ok(u),
        Err(_) => Err(ValidationError::new(format!(
            "Invalid ID {}, expected ULID",
            id
        ))),
    }
}
