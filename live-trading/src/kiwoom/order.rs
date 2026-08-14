use serde::{Deserialize, Serialize};

use super::client::KiwoomClient;

#[derive(Debug, Serialize)]
pub struct OrderRequest {
    pub dmst_stex_tp: String,
    pub stk_cd: String,
    pub ord_qty: String,
    pub ord_uv: String,
    pub trde_tp: String,
    pub cond_uv: String,
}

#[derive(Debug, Deserialize)]
pub struct OrderResponse {
    pub ord_no: Option<String>,
    pub dmst_stex_tp: Option<String>,
    pub return_code: Option<i32>,
    pub return_msg: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum OrderType {
    Market,
    Limit,
}

impl OrderType {
    pub fn to_code(&self) -> &'static str {
        match self {
            OrderType::Market => "3",
            OrderType::Limit => "0",
        }
    }
}

pub struct OrderApi {
    client: KiwoomClient,
    endpoint: String,
}

impl OrderApi {
    pub fn new(client: KiwoomClient) -> Self {
        Self {
            client,
            endpoint: "/api/dostk/ordr".to_string(),
        }
    }

    pub async fn buy(
        &self,
        token: &str,
        stock_code: &str,
        qty: i32,
        price: Option<i32>,
        order_type: OrderType,
    ) -> Result<OrderResponse, Box<dyn std::error::Error + Send + Sync>> {
        let ord_uv = match order_type {
            OrderType::Market => String::new(),
            OrderType::Limit => price.map(|p| p.to_string()).unwrap_or_default(),
        };

        let payload = OrderRequest {
            dmst_stex_tp: "KRX".to_string(),
            stk_cd: stock_code.to_string(),
            ord_qty: qty.to_string(),
            ord_uv,
            trde_tp: order_type.to_code().to_string(),
            cond_uv: String::new(),
        };

        self.client
            .post(&self.endpoint, &payload, vec![
                ("authorization", &format!("Bearer {}", token)),
                ("api-id", "kt10000"),
            ])
            .await
            .map_err(|e| e.into())
    }

    pub async fn sell(
        &self,
        token: &str,
        stock_code: &str,
        qty: i32,
        price: Option<i32>,
        order_type: OrderType,
    ) -> Result<OrderResponse, Box<dyn std::error::Error + Send + Sync>> {
        let ord_uv = match order_type {
            OrderType::Market => String::new(),
            OrderType::Limit => price.map(|p| p.to_string()).unwrap_or_default(),
        };

        let payload = OrderRequest {
            dmst_stex_tp: "KRX".to_string(),
            stk_cd: stock_code.to_string(),
            ord_qty: qty.to_string(),
            ord_uv,
            trde_tp: order_type.to_code().to_string(),
            cond_uv: String::new(),
        };

        self.client
            .post(&self.endpoint, &payload, vec![
                ("authorization", &format!("Bearer {}", token)),
                ("api-id", "kt10001"),
            ])
            .await
            .map_err(|e| e.into())
    }

    pub async fn market_buy(
        &self,
        token: &str,
        stock_code: &str,
        qty: i32,
    ) -> Result<OrderResponse, Box<dyn std::error::Error + Send + Sync>> {
        self.buy(token, stock_code, qty, None, OrderType::Market).await
    }

    pub async fn market_sell(
        &self,
        token: &str,
        stock_code: &str,
        qty: i32,
    ) -> Result<OrderResponse, Box<dyn std::error::Error + Send + Sync>> {
        self.sell(token, stock_code, qty, None, OrderType::Market).await
    }
}
