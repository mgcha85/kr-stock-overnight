use serde::{Deserialize, Serialize};

use super::client::KiwoomClient;

#[derive(Debug, Serialize)]
pub struct AccountRequest {
    pub canm: String,
    pub qry_tp: String,
    pub dmst_stex_tp: String,
}

#[derive(Debug, Deserialize)]
pub struct AssetResponse {
    pub return_code: Option<i32>,
    pub return_msg: Option<String>,
    pub entr: Option<String>,
    pub tot_pur_amt: Option<String>,
    pub tot_evlt_amt: Option<String>,
    pub prsm_dpst_aset_amt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DepositResponse {
    pub return_code: Option<i32>,
    pub return_msg: Option<String>,
    #[serde(rename = "entr")]
    pub deposit: Option<String>,
    #[serde(rename = "ord_alow_amt")]
    pub orderable_cash: Option<String>,
    #[serde(rename = "pymn_alow_amt")]
    pub withdrawable_cash: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AccountEvalResponse {
    pub return_code: Option<i32>,
    pub return_msg: Option<String>,
    #[serde(rename = "stk_acnt_evlt_prst", default)]
    pub holdings: Vec<HoldingItem>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct HoldingItem {
    pub stk_cd: Option<String>,
    pub stk_nm: Option<String>,
    pub rmnd_qty: Option<String>,
    pub avg_prc: Option<String>,
    pub cur_prc: Option<String>,
    pub evlt_pl: Option<String>,
    pub pchs_amt: Option<String>,
    pub evlt_amt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TradeHistoryRequest {
    pub canm: String,
    pub strt_dt: String,
    pub end_dt: String,
    pub dmst_stex_tp: String,
    pub tp: String,
    pub gds_tp: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TradeHistoryItem {
    pub trde_dt: Option<String>,
    pub cntr_dt: Option<String>,
    pub proc_tm: Option<String>,
    pub stk_cd: Option<String>,
    pub stk_nm: Option<String>,
    pub io_tp: Option<String>,
    pub io_tp_nm: Option<String>,
    pub trde_qty_jwa_cnt: Option<String>,
    pub trde_unit: Option<String>,
    pub trde_amt: Option<String>,
    pub exct_amt: Option<String>,
    pub cmsn: Option<String>,
    pub trde_agri_tax: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TradeHistoryResponse {
    #[serde(rename = "trst_ovrl_trde_prps_array", default)]
    items: Vec<TradeHistoryItem>,
    return_code: Option<i32>,
    return_msg: Option<String>,
}

pub struct AccountService {
    client: KiwoomClient,
    endpoint: String,
}

impl AccountService {
    pub fn new(client: KiwoomClient) -> Self {
        Self {
            client,
            endpoint: "/api/dostk/acnt".to_string(),
        }
    }

    pub async fn get_asset(&self, token: &str, account_num: &str) -> Result<AssetResponse, Box<dyn std::error::Error + Send + Sync>> {
        let payload = AccountRequest {
            canm: account_num.to_string(),
            qry_tp: "0".to_string(),
            dmst_stex_tp: "KRX".to_string(),
        };

        let response = self.client
            .post_json_with_headers(&self.endpoint, &payload, vec![
                ("authorization", &format!("Bearer {}", token)),
                ("api-id", "kt00003"),
            ])
            .await?;
            
        let (_headers, json) = response;
        tracing::debug!("Raw asset response: {:?}", json);
        
        serde_json::from_value(json).map_err(|e| e.into())
    }

    pub async fn get_holdings(&self, token: &str, account_num: &str) -> Result<Vec<HoldingItem>, Box<dyn std::error::Error + Send + Sync>> {
        let payload = AccountRequest {
            canm: account_num.to_string(),
            qry_tp: "1".to_string(),
            dmst_stex_tp: "KRX".to_string(),
        };

        let response_v: serde_json::Value = self.client
            .post_json_with_headers(&self.endpoint, &payload, vec![
                ("authorization", &format!("Bearer {}", token)),
                ("api-id", "kt00004"),
            ])
            .await
            .map(|(_, v)| v)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        tracing::debug!("Raw holdings response: {:?}", response_v);

        let response: AccountEvalResponse = serde_json::from_value(response_v.clone()).map_err(|e| {
            tracing::error!("Failed to parse holdings JSON: {}. JSON: {:?}", e, response_v);
            Box::new(e) as Box<dyn std::error::Error + Send + Sync>
        })?;

        Ok(response.holdings)
    }

    pub async fn get_deposit(&self, token: &str, account_num: &str) -> Result<DepositResponse, Box<dyn std::error::Error + Send + Sync>> {
        let payload = AccountRequest {
            canm: account_num.to_string(),
            qry_tp: "0".to_string(),
            dmst_stex_tp: "KRX".to_string(),
        };

        let response = self.client
            .post_json_with_headers(&self.endpoint, &payload, vec![
                ("authorization", &format!("Bearer {}", token)),
                ("api-id", "kt00001"),
            ])
            .await?;
            
        let (_headers, json) = response;
        tracing::info!("Raw deposit response (kt00001): {:?}", json);
        
        serde_json::from_value(json).map_err(|e| e.into())
    }

    pub async fn get_cash_balance(&self, token: &str, account_num: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let deposit = self.get_deposit(token, account_num).await?;
        
        // ord_alow_amt (주문가능금액) > entr (예수금) > pymn_alow_amt (출금가능금액)
        let cash = deposit.orderable_cash
            .as_ref()
            .or(deposit.deposit.as_ref())
            .or(deposit.withdrawable_cash.as_ref())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(0.0);
        
        Ok(cash)
    }

    pub async fn get_trade_history(
        &self,
        token: &str,
        account_num: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<Vec<TradeHistoryItem>, Box<dyn std::error::Error + Send + Sync>> {
        let payload = TradeHistoryRequest {
            canm: account_num.to_string(),
            strt_dt: start_date.to_string(),
            end_dt: end_date.to_string(),
            dmst_stex_tp: "KRX".to_string(),
            tp: "0".to_string(),
            gds_tp: "0".to_string(),
        };

        let response: TradeHistoryResponse = self
            .client
            .post(
                &self.endpoint,
                &payload,
                vec![
                    ("authorization", &format!("Bearer {}", token)),
                    ("api-id", "kt00015"),
                ],
            )
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        if response.return_code.unwrap_or(0) != 0 {
            let msg = response
                .return_msg
                .unwrap_or_else(|| "Unknown kt00015 error".to_string());
            return Err(msg.into());
        }

        Ok(response.items)
    }

    pub async fn get_total_asset(&self, token: &str, account_num: &str) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let asset = self.get_asset(token, account_num).await?;
        let entr = asset.prsm_dpst_aset_amt
            .as_ref()
            .or(asset.entr.as_ref())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let tot_pur_amt = asset.tot_pur_amt
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        Ok(entr + tot_pur_amt)
    }
}
