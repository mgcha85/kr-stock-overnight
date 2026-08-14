use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use chrono::{DateTime, Utc, Duration};

use super::client::KiwoomClient;

const TOKEN_EXPIRY_HOURS: i64 = 12;

#[derive(Debug, Serialize, Deserialize)]
struct TokenData {
    token: String,
    issued_at: DateTime<Utc>,
}

impl TokenData {
    fn new(token: String) -> Self {
        Self {
            token,
            issued_at: Utc::now(),
        }
    }

    fn is_expired(&self) -> bool {
        Utc::now() > self.issued_at + Duration::hours(TOKEN_EXPIRY_HOURS)
    }
}

/// OAuth request payload
#[derive(Debug, Serialize)]
struct OAuthRequest {
    grant_type: String,
    appkey: String,
    secretkey: String,
}

/// OAuth response
#[derive(Debug, Deserialize)]
pub struct OAuthResponse {
    pub token: Option<String>,
    pub return_code: Option<i32>,
    pub return_msg: Option<String>,
}

/// Token revoke request
#[derive(Debug, Serialize)]
struct RevokeRequest {
    appkey: String,
    secretkey: String,
    token: String,
}

/// Authentication manager for Kiwoom API
pub struct KiwoomAuth {
    client: KiwoomClient,
    app_key: String,
    secret_key: String,
    token_file: String,
}

impl KiwoomAuth {
    pub fn new(client: KiwoomClient, app_key: String, secret_key: String) -> Self {
        Self {
            client,
            app_key,
            secret_key,
            token_file: "access_token.txt".to_string(),
        }
    }

    /// Create auth manager from environment
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        let app_key = std::env::var("APP_KEY").expect("APP_KEY not set");
        let secret_key = std::env::var("SECRET_KEY").expect("SECRET_KEY not set");
        
        Self {
            client: KiwoomClient::from_env(),
            app_key,
            secret_key,
            token_file: "access_token.txt".to_string(),
        }
    }

    /// Create auth manager with custom token file path
    pub fn with_token_file(mut self, path: &str) -> Self {
        self.token_file = path.to_string();
        self
    }

    /// Get WebSocket URL from environment
    pub fn ws_url() -> String {
        dotenvy::dotenv().ok();
        std::env::var("WS_URL")
            .unwrap_or_else(|_| "wss://api.kiwoom.com:10000/api/dostk/websocket".to_string())
    }

    /// Request new access token from Kiwoom API
    pub async fn request_token(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let payload = OAuthRequest {
            grant_type: "client_credentials".to_string(),
            appkey: self.app_key.clone(),
            secretkey: self.secret_key.clone(),
        };

        let response: OAuthResponse = self.client
            .post("/oauth2/token", &payload, vec![])
            .await?;

        if let Some(code) = response.return_code {
            if code == 3 {
                return Err(format!("Auth error: {:?}", response.return_msg).into());
            }
        }

        response.token.ok_or_else(|| "No token in response".into())
    }

    /// Get token and save to file with timestamp
    pub async fn set_access_token(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let token = self.request_token().await?;
        let token_data = TokenData::new(token.clone());
        let json = serde_json::to_string(&token_data)?;
        fs::write(&self.token_file, &json)?;
        tracing::info!("Token saved to {} (expires in {} hours)", self.token_file, TOKEN_EXPIRY_HOURS);
        Ok(token)
    }

    /// Load token from file, returns None if expired or not found
    pub fn get_access_token(&self) -> Option<String> {
        let path = Path::new(&self.token_file);
        if !path.exists() {
            return None;
        }
        
        let content = fs::read_to_string(path).ok()?;
        
        if let Ok(token_data) = serde_json::from_str::<TokenData>(&content) {
            if token_data.is_expired() {
                tracing::info!("Token expired, will refresh");
                return None;
            }
            return Some(token_data.token);
        }
        
        // Fallback: old format (plain token string) - treat as expired
        tracing::info!("Old token format detected, will refresh");
        None
    }

    /// Get or refresh token
    pub async fn ensure_token(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(token) = self.get_access_token() {
            Ok(token)
        } else {
            self.set_access_token().await
        }
    }

    /// Force refresh token (useful when token expired)
    pub async fn refresh_token(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        tracing::info!("Refreshing access token...");
        self.set_access_token().await
    }

    /// Revoke token
    pub async fn revoke_token(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let token = match self.get_access_token() {
            Some(t) => t,
            None => {
                tracing::warn!("No token to revoke");
                return Ok(());
            }
        };

        let payload = RevokeRequest {
            appkey: self.app_key.clone(),
            secretkey: self.secret_key.clone(),
            token,
        };

        let response: serde_json::Value = self.client
            .post("/oauth2/revoke", &payload, vec![])
            .await?;

        if response.get("return_code").and_then(|v| v.as_i64()) == Some(0) {
            tracing::info!("Token revoked successfully");
            let _ = fs::remove_file(&self.token_file);
        } else {
            tracing::warn!("Token revoke failed: {:?}", response);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_creation() {
        std::env::set_var("APP_KEY", "test_key");
        std::env::set_var("SECRET_KEY", "test_secret");
        std::env::set_var("APP_DOMAIN", "https://api.kiwoom.com");
        
        let auth = KiwoomAuth::from_env();
        assert_eq!(auth.app_key, "test_key");
    }
}
