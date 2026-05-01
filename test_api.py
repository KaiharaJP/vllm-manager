#!/usr/bin/env python3
"""FastAPI サーバーの API テスト"""

import httpx
import json
import time

API_BASE = "http://localhost:8765"

def test_api():
    print("=== API テスト ===")
    
    # 待機
    time.sleep(3)
    
    # ステータス取得
    print("\n--- GET /api/status ---")
    resp = httpx.get(f"{API_BASE}/api/status")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")
    
    # モデルリスト取得
    print("\n--- GET /api/models ---")
    resp = httpx.get(f"{API_BASE}/api/models")
    print(f"Status: {resp.status_code}")
    models = resp.json()
    print(f"モデル数: {len(models)}")
    for m in models[:3]:
        print(f"  - {m['name']} ({m['size']})")
    
    # コンテキストプリセット取得
    print("\n--- GET /api/context-presets ---")
    resp = httpx.get(f"{API_BASE}/api/context-presets")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")
    
    # 設定取得
    print("\n--- GET /api/config ---")
    resp = httpx.get(f"{API_BASE}/api/config")
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(resp.json(), indent=2)}")
    
    # ログ取得
    print("\n--- GET /api/log ---")
    resp = httpx.get(f"{API_BASE}/api/log")
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()['log'][:200]}...")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_api()
