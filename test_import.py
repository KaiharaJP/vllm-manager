#!/usr/bin/env python3
"""FastAPI アプリのインポートテスト"""

import sys
sys.path.insert(0, "/workspace/vllm-manager")

def test_imports():
    print("=== インポートテスト ===")
    
    # サーバーマネージャー
    from app.server_manager import (
        get_status, load_config, get_available_models, get_context_presets
    )
    print(f"[OK] server_manager")
    print(f"  モデル数: {len(get_available_models())}")
    print(f"  コンテキストプリセット: {[p['label'] for p in get_context_presets()]}")
    print(f"  デフォルト設定: {load_config()}")
    print(f"  サーバー状態: {get_status()}")
    
    # メトリクススクラッパー
    from app.metrics_scraper import MetricsScraper
    print(f"[OK] metrics_scraper")
    
    # FastAPI アプリ
    from app.main import app
    print(f"[OK] FastAPI app")
    print(f"  API ルート数: {len(app.routes)}")
    
    # ルート一覧
    print("\n=== API エンドポイント ===")
    for route in app.routes:
        if hasattr(route, "methods"):
            methods = ",".join(route.methods)
            print(f"  {methods} {route.path}")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_imports()
