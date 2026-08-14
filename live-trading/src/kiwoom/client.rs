use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Kiwoom REST API client
#[derive(Clone)]
pub struct KiwoomClient {
    http: Client,
    base_url: String,
}

impl KiwoomClient {
    /// Create a new Kiwoom API client
    pub fn new(base_url: &str) -> Self {
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| Client::new()),
            base_url: base_url.to_string(),
        }
    }

    /// Create client from environment
    pub fn from_env() -> Self {
        dotenvy::dotenv().ok();
        let base_url = std::env::var("APP_DOMAIN")
            .unwrap_or_else(|_| "https://api.kiwoom.com".to_string());
        Self::new(&base_url)
    }

    /// Get base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// POST request with JSON body
    pub async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
        headers: Vec<(&str, &str)>,
    ) -> Result<R, reqwest::Error> {
        let url = format!("{}{}", self.base_url, endpoint);
        
        let mut request = self.http
            .post(&url)
            .header("Content-Type", "application/json;charset=UTF-8")
            .json(body);

        for (key, value) in headers {
            request = request.header(key, value);
        }

        let response = request.send().await?;
        response.json::<R>().await
    }

    /// POST request returning raw JSON
    pub async fn post_json<T: Serialize>(
        &self,
        endpoint: &str,
        body: &T,
        headers: Vec<(&str, &str)>,
    ) -> Result<serde_json::Value, reqwest::Error> {
        self.post(endpoint, body, headers).await
    }

    /// POST request returning headers and raw JSON
    pub async fn post_json_with_headers<T: Serialize>(
        &self,
        endpoint: &str,
        body: &T,
        headers: Vec<(&str, &str)>,
    ) -> Result<(reqwest::header::HeaderMap, serde_json::Value), reqwest::Error> {
        let url = format!("{}{}", self.base_url, endpoint);
        
        let mut request = self.http
            .post(&url)
            .header("Content-Type", "application/json;charset=UTF-8")
            .json(body);

        for (key, value) in headers {
            request = request.header(key, value);
        }

        let response = request.send().await?;
        let headers = response.headers().clone();
        let json = response.json::<serde_json::Value>().await?;
        
        Ok((headers, json))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = KiwoomClient::new("https://api.kiwoom.com");
        assert_eq!(client.base_url, "https://api.kiwoom.com");
    }
}
