"""
Kiwoom OpenAPI REST API Server Daemon Mock / Bridge
---------------------------------------------------
Serves Kiwoom HTS Condition Search API on http://localhost:5000/api/condition
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("kiwoom_rest_api")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Default 25 HTS "종가베팅" Condition Search Tickers for 2026-08-12
# Includes Top-3: OCI홀딩스(010060), 현대해상(001450), 산일전기(062040)
HTS_JONGGA_25_CODES = [
    "010060", "001450", "062040", "005930", "006400",
    "009830", "010170", "010950", "035720", "036930",
    "042700", "043260", "064760", "066970", "067310",
    "068270", "080220", "086520", "096770", "103590",
    "131290", "131970", "181710", "196170", "214450"
]

class KiwoomAPIRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/api/condition":
            params = parse_qs(parsed_path.query)
            condition_name = params.get("name", ["종가베팅"])[0]

            response_data = {
                "status": "success",
                "condition_name": condition_name,
                "count": len(HTS_JONGGA_25_CODES),
                "codes": HTS_JONGGA_25_CODES,
                "source": "mock",
                "as_of": "2026-08-12",
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        logger.info(f"Kiwoom REST API Call: {args[0]}")

def run_server(port: int = 5000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, KiwoomAPIRequestHandler)
    logger.info(f"Kiwoom OpenAPI REST Server Daemon running on http://localhost:{port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Kiwoom REST Server shutting down...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
