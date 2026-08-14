use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message, tungstenite::client::IntoClientRequest};
use std::collections::HashMap;
use serde_json::Value;

pub async fn fetch_condition_stocks(
    condition_names: &[String],
    token: &str,
    ws_url: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    if condition_names.is_empty() {
        tracing::info!("No condition names provided. Returning empty list.");
        return Ok(Vec::new());
    }

    tracing::info!("Connecting to WebSocket for condition search: {}", ws_url);
    let request = ws_url.into_client_request()?;

    let (ws_stream, _) = connect_async(request).await?;
    let (mut write, mut read) = ws_stream.split();

    let login_req = serde_json::json!({
        "trnm": "LOGIN",
        "token": token
    });
    tracing::info!("Sending LOGIN");
    write.send(Message::Text(login_req.to_string())).await?;

    let mut is_logged_in = false;
    let mut timeout = 0;
    while let Some(msg) = read.next().await {
        if let Ok(Message::Text(text)) = msg {
            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                if json.get("trnm").and_then(|v| v.as_str()) == Some("PING") {
                    let _ = write.send(Message::Text(text)).await;
                    continue;
                }
                if json.get("trnm").and_then(|v| v.as_str()) == Some("LOGIN") {
                    if json.get("return_code").and_then(|v| v.as_i64()).unwrap_or(-1) == 0 {
                        tracing::info!("LOGIN success");
                        is_logged_in = true;
                        break;
                    } else {
                        tracing::error!("LOGIN failed: {:?}", json);
                        return Ok(Vec::new());
                    }
                }
            }
        }
        timeout += 1;
        if timeout > 20 { break; }
    }

    if !is_logged_in {
        tracing::error!("WebSocket LOGIN timeout");
        return Ok(Vec::new());
    }

    let cnsrlst_req = serde_json::json!({ "trnm": "CNSRLST" });
    write.send(Message::Text(cnsrlst_req.to_string())).await?;

    let mut condition_map: HashMap<String, String> = HashMap::new();
    timeout = 0;
    while let Some(msg) = read.next().await {
        if let Ok(Message::Text(text)) = msg {
            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                if json.get("trnm").and_then(|v| v.as_str()) == Some("PING") {
                    let _ = write.send(Message::Text(text)).await;
                    continue;
                }
                if json.get("trnm").and_then(|v| v.as_str()) == Some("CNSRLST") {
                    if let Some(data) = json.get("data").and_then(|v| v.as_array()) {
                        for item in data {
                            if let Some(arr) = item.as_array() {
                                if arr.len() >= 2 {
                                    let seq = arr[0].as_str().unwrap_or("").to_string();
                                    let name = arr[1].as_str().unwrap_or("").to_string();
                                    condition_map.insert(name, seq);
                                }
                            }
                        }
                    }
                    tracing::info!("Loaded {} conditions from server", condition_map.len());
                    break;
                }
            }
        }
        timeout += 1;
        if timeout > 20 { break; }
    }

    let mut all_stocks: Vec<String> = Vec::new();

    for name in condition_names {
        if let Some(seq) = condition_map.get(name) {
            tracing::info!("Condition '{}' found (Seq: {}). Requesting CNSRREQ...", name, seq);

            let mut cont_yn = "N".to_string();
            let mut next_key = "".to_string();

            loop {
                let req_payload = serde_json::json!({
                    "trnm": "CNSRREQ",
                    "seq": seq,
                    "search_type": "0",
                    "stex_tp": "K",
                    "cont_yn": cont_yn,
                    "next_key": next_key
                });
                write.send(Message::Text(req_payload.to_string())).await?;

                let mut got_response = false;
                timeout = 0;
                while let Some(msg) = read.next().await {
                    if let Ok(Message::Text(text)) = msg {
                        if let Ok(json) = serde_json::from_str::<Value>(&text) {
                            if json.get("trnm").and_then(|v| v.as_str()) == Some("PING") {
                                let _ = write.send(Message::Text(text)).await;
                                continue;
                            }
                            if json.get("trnm").and_then(|v| v.as_str()) == Some("CNSRREQ") {
                                got_response = true;
                                if let Some(data) = json.get("data").and_then(|v| v.as_array()) {
                                    let mut num_added = 0;
                                    for item in data {
                                        let code = item.get("9001").or(item.get("code")).or(item.get("stk_cd"))
                                            .and_then(|v| v.as_str());
                                        if let Some(c) = code {
                                            let clean_code = c.trim_start_matches(|ch: char| ch.is_alphabetic())
                                                .split('.').next().unwrap_or(c);
                                            if !all_stocks.contains(&clean_code.to_string()) {
                                                all_stocks.push(clean_code.to_string());
                                                num_added += 1;
                                            }
                                        }
                                    }
                                    tracing::info!("Extracted {} stocks for condition '{}'", num_added, name);
                                }

                                cont_yn = json.get("cont_yn").or(json.get("cont-yn")).and_then(|v| v.as_str()).unwrap_or("N").to_uppercase();
                                next_key = json.get("next_key").or(json.get("next-key")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                                break;
                            }
                        }
                    }
                    timeout += 1;
                    if timeout > 20 { break; }
                }

                if !got_response || cont_yn != "Y" || next_key.is_empty() {
                    break;
                }
            }
        } else {
            tracing::warn!("Requested condition '{}' not found on the server.", name);
        }
    }

    tracing::info!("Closing WebSocket connection.");
    let _ = write.send(Message::Close(None)).await;

    Ok(all_stocks)
}

pub async fn list_conditions(
    token: &str,
    ws_url: &str,
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error + Send + Sync>> {
    tracing::info!("Connecting to WebSocket to list conditions: {}", ws_url);
    let request = ws_url.into_client_request()?;

    let (ws_stream, _) = connect_async(request).await?;
    let (mut write, mut read) = ws_stream.split();

    let login_req = serde_json::json!({
        "trnm": "LOGIN",
        "token": token
    });
    write.send(Message::Text(login_req.to_string())).await?;

    let mut is_logged_in = false;
    let mut timeout = 0;
    while let Some(msg) = read.next().await {
        if let Ok(Message::Text(text)) = msg {
            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                if json.get("trnm").and_then(|v| v.as_str()) == Some("PING") {
                    let _ = write.send(Message::Text(text)).await;
                    continue;
                }
                if json.get("trnm").and_then(|v| v.as_str()) == Some("LOGIN") {
                    if json.get("return_code").and_then(|v| v.as_i64()).unwrap_or(-1) == 0 {
                        is_logged_in = true;
                        break;
                    }
                }
            }
        }
        timeout += 1;
        if timeout > 20 { break; }
    }

    if !is_logged_in {
        return Err("WebSocket LOGIN timeout".into());
    }

    let cnsrlst_req = serde_json::json!({ "trnm": "CNSRLST" });
    write.send(Message::Text(cnsrlst_req.to_string())).await?;

    let mut conditions: Vec<(String, String)> = Vec::new();
    timeout = 0;
    while let Some(msg) = read.next().await {
        if let Ok(Message::Text(text)) = msg {
            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                if json.get("trnm").and_then(|v| v.as_str()) == Some("PING") {
                    let _ = write.send(Message::Text(text)).await;
                    continue;
                }
                if json.get("trnm").and_then(|v| v.as_str()) == Some("CNSRLST") {
                    if let Some(data) = json.get("data").and_then(|v| v.as_array()) {
                        for item in data {
                            if let Some(arr) = item.as_array() {
                                if arr.len() >= 2 {
                                    let seq = arr[0].as_str().unwrap_or("").to_string();
                                    let name = arr[1].as_str().unwrap_or("").to_string();
                                    conditions.push((seq, name));
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
        timeout += 1;
        if timeout > 20 { break; }
    }

    let _ = write.send(Message::Close(None)).await;
    Ok(conditions)
}
