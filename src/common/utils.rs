use crate::definitions::errors::ValidationError;
use ulid::Ulid;

pub fn parse_ulid(id: &str) -> Result<Ulid, ValidationError> {
    id.parse::<Ulid>()
        .map_err(|_| ValidationError::new(format!("Invalid ID {}, expected ULID", id)))
}
