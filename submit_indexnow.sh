#!/bin/bash
# =============================================================================
# IndexNow Submission Script for TEP-GNSS (Both Manuscripts)
# =============================================================================
# This script notifies search engines (Bing, Yandex, etc.) when papers are updated.
# Run this after pushing changes to trigger immediate re-crawling.
#
# Usage: ./submit_indexnow.sh
# =============================================================================

# Configuration
HOST="matthewsmawfield.github.io"
KEY="621a69ca2bd75c0778c9658dc1a62f0a"
KEY_LOCATION="https://matthewsmawfield.github.io/TEP-GNSS/${KEY}.txt"

# URLs to index (Main manuscript)
MAIN_URL="https://matthewsmawfield.github.io/TEP-GNSS/"
MAIN_PDF="https://matthewsmawfield.github.io/TEP-GNSS/public/docs/Smawfield_2025_GlobalTimeEchoes_v0.23_Jaipur.pdf"

# URLs to index (Paper 2 - now in separate repo TEP-GNSS-II)
LONGSPAN_URL="https://matthewsmawfield.github.io/TEP-GNSS-II/"
LONGSPAN_PDF="https://matthewsmawfield.github.io/TEP-GNSS-II/public/docs/Smawfield_2025_GlobalTimeEchoes_25Year_v0.16_Cairo.pdf"

# JSON Payload
JSON_PAYLOAD=$(cat <<EOF
{
  "host": "$HOST",
  "key": "$KEY",
  "keyLocation": "$KEY_LOCATION",
  "urlList": [
    "$MAIN_URL",
    "$MAIN_PDF",
    "$LONGSPAN_URL",
    "$LONGSPAN_PDF"
  ]
}
EOF
)

echo "=============================================="
echo "IndexNow Submission for TEP-GNSS"
echo "=============================================="
echo ""
echo "Host: $HOST"
echo "Key Location: $KEY_LOCATION"
echo ""
echo "URLs to index:"
echo "  [Main Manuscript]"
echo "  - $MAIN_URL"
echo "  - $MAIN_PDF"
echo "  [CODE Longspan Manuscript]"
echo "  - $LONGSPAN_URL"
echo "  - $LONGSPAN_PDF"
echo ""
echo "Submitting to IndexNow API..."
echo ""

# Submit to IndexNow (shared across Bing, Yandex, and other participating engines)
curl -s -X POST "https://api.indexnow.org/indexnow" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d "$JSON_PAYLOAD" \
     -w "\nHTTP Status: %{http_code}\n"

echo ""
echo "=============================================="
echo "Submission complete!"
echo ""
echo "HTTP 200 = Success (URLs received by search engines)"
echo "HTTP 202 = Accepted (URLs queued for processing)"
echo "HTTP 400 = Bad Request (check JSON payload)"
echo "HTTP 403 = Forbidden (key validation failed)"
echo "HTTP 422 = Unprocessable (URLs don't match key location)"
echo "=============================================="
