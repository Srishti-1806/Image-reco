#!/usr/bin/env bash
# Run on the CORE server. Allows only the edge subnet to reach the gateway.
set -euo pipefail
EDGE_SUBNET="${1:?e.g. 192.168.50.0/24}"
iptables -P INPUT DROP
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 8443 -s "$EDGE_SUBNET" -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -s "$EDGE_SUBNET" -j ACCEPT   # admin SSH, adjust
echo "Rules applied. Persist them with iptables-persistent / nftables."
