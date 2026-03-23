const _ENV_PREFIX: &str = "ATLAS_";

fn get_setting_from_env_or_default(key: &str, default: &str) -> String {
    std::env::var(format!("{}{}", _ENV_PREFIX, key)).unwrap_or_else(|_| default.to_string())
}

pub struct Settings;

impl Settings {
    pub fn port(&self) -> String {
        get_setting_from_env_or_default("PORT", "8600")
    }

    pub fn host(&self) -> String {
        get_setting_from_env_or_default("HOST", "0.0.0.0")
    }

    pub fn log_level(&self) -> String {
        get_setting_from_env_or_default("LOG_LEVEL", "info")
    }

    pub fn data_dir(&self) -> String {
        get_setting_from_env_or_default("DATA_DIR", "./data")
    }

    pub fn flush_interval(&self) -> u64 {
        get_setting_from_env_or_default("FLUSH_INTERVAL", "60")
            .parse()
            .unwrap_or(60)
    }
}

pub static SETTINGS: Settings = Settings;
