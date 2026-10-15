#!/usr/bin/env bash
# Usage:  ./gen_certs.sh init <server-hostname>
#         ./gen_certs.sh device <device-name>      (CN must match ALLOWED_DEVICES)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/certs"; mkdir -p "$DIR"; cd "$DIR"
case "${1:-}" in
  init)
    HOST="${2:?server hostname}"
    openssl genrsa -out ca.key 4096
    openssl req -x509 -new -key ca.key -sha256 -days 1825 -subj "/CN=face-recog-CA" -out ca.crt
    openssl genrsa -out server.key 2048
    openssl req -new -key server.key -subj "/CN=$HOST" -out server.csr
    printf "subjectAltName=DNS:%s\n" "$HOST" > san.ext
    openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
      -out server.crt -days 365 -sha256 -extfile san.ext
    chmod 600 *.key ;;
  device)
    NAME="${2:?device name}"
    openssl genrsa -out "$NAME.key" 2048
    openssl req -new -key "$NAME.key" -subj "/CN=$NAME" -out "$NAME.csr"
    openssl x509 -req -in "$NAME.csr" -CA ca.crt -CAkey ca.key -CAcreateserial \
      -out "$NAME.crt" -days 365 -sha256
    chmod 600 "$NAME.key" ;;
  *) echo "usage: $0 init <host> | device <name>"; exit 1 ;;
esac
