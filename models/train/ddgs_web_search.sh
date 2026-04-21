#!/bin/bash
port="${1:-8280}"
python "$(dirname "$0")/ddgs_web_search.py" --port "$port" --topk 5 --backend "google,brave,bing,wikipedia,grokipedia"