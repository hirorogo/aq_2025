# ☕ Caffeine - Mac学習時のスリープ防止ツール

## 🎯 Caffeineとは
Caffeineは、MacBookがスリープモードに入るのを防ぐアプリです。
じゃんけんAIの学習中に画面を閉じてもプロセスが継続されるようになります。

## 🚀 使い方

### 1. アプリの起動
```bash
# アプリケーションフォルダから起動
open /Applications/Caffeine.app
```

または、Launchpadから「Caffeine」を検索して起動

### 2. 状態確認
メニューバー（画面上部）にコーヒーカップのアイコンが表示されます：

- ☕ **空のカップ**：スリープ防止OFF（通常モード）
- ☕ **満たされたカップ**：スリープ防止ON（学習時に使用）

### 3. 基本操作

#### スリープ防止をONにする
1. メニューバーのコーヒーカップアイコンをクリック
2. 「Turn Caffeine On」を選択
3. アイコンが満たされたカップに変わることを確認

#### スリープ防止をOFFにする  
1. メニューバーのコーヒーカップアイコンをクリック
2. 「Turn Caffeine Off」を選択
3. アイコンが空のカップに変わることを確認

## 🤖 じゃんけんAI学習時の使用手順

### 学習開始前
```bash
# 1. Caffeineを起動
open /Applications/Caffeine.app

# 2. プロジェクトフォルダに移動
cd /Users/hiro/Documents/aq_2025

# 3. 仮想環境を有効化
source venv/bin/activate
```

### Caffeineを有効化
1. メニューバーのコーヒーカップをクリック
2. 「Turn Caffeine On」を選択
3. アイコンが満たされたカップになったことを確認

### 学習実行
```bash
# 安全版で学習開始
python3 janken_train_new_safe.py
```

### 学習完了後
1. 学習が完了したら、Caffeineをクリック
2. 「Turn Caffeine Off」を選択
3. 通常のスリープ設定に戻る

## ⚙️ 高度な設定

### 1. 自動起動設定
```bash
# ログイン時に自動でCaffeineを起動
# システム設定 > 一般 > ログイン項目 > Caffeineを追加
```

### 2. 時間指定モード
メニューバーから以下を選択可能：
- **Indefinitely**：手動でOFFにするまで継続
- **For 5 minutes**：5分間のみスリープ防止
- **For 10 minutes**：10分間のみスリープ防止
- **For 15 minutes**：15分間のみスリープ防止
- **For 30 minutes**：30分間のみスリープ防止
- **For 1 hour**：1時間のみスリープ防止

### 3. 学習時間に応じた設定
```
学習予想時間に応じてCaffeineの時間を設定：

🕐 短時間学習（10-20分）→ 30分設定
🕑 中時間学習（20-40分）→ 1時間設定  
🕒 長時間学習（1時間以上）→ Indefinitely設定
```

## 🛡️ 安全な使用方法

### ✅ やるべきこと
- 学習開始前にCaffeineをON
- 学習完了後にCaffeineをOFF
- 電源アダプターを接続
- バッテリー残量を確認（80%以上推奨）

### ❌ 注意すること
- Caffeineを付けたまま長時間放置しない
- バッテリーのみで長時間使用しない
- 他の重要な作業中は使用を控える

## 🔋 バッテリー管理

### Caffeeine使用時のバッテリー確認
```bash
# バッテリー状況確認
pmset -g batt

# 充電状況を監視
while true; do
  echo "$(date): $(pmset -g batt | grep -o '[0-9]*%')"
  sleep 300  # 5分ごとにチェック
done
```

### 電源設定との組み合わせ
```bash
# Caffeine使用時の推奨設定
sudo pmset -c displaysleep 0    # ディスプレイスリープ無効
sudo pmset -c sleep 0           # システムスリープ無効

# 学習完了後に戻す
sudo pmset -c displaysleep 10   # 10分後にディスプレイスリープ
sudo pmset -c sleep 30          # 30分後にシステムスリープ
```

## 💤 夜間学習時の完全ガイド

### 寝る前のチェックリスト
- [ ] 電源アダプター接続確認
- [ ] Caffeineアプリ起動確認
- [ ] CaffeineをON（Indefinitelyモード）
- [ ] 学習スクリプト実行
- [ ] エラーが発生していないか確認r
- [ ] 予想完了時間をメモ

### 朝起きた時の確認
- [ ] 学習が正常完了したか確認
- [ ] モデルファイルが保存されているか確認
- [ ] CaffeineをOFFに戻す
- [ ] 省エネルギー設定を元に戻す

## 🔧 トラブルシューティング

### Caffeineが動作しない場合
```bash
# Caffeineプロセス確認
ps aux | grep -i caffeine

# 再起動
killall Caffeine
open /Applications/Caffeine.app
```

### システムがスリープしてしまう場合
1. **システム設定を確認**
   - 省エネルギー設定
   - スクリーンセーバー設定

2. **Caffeineの設定を確認**
   - アイコンが満たされているか
   - Indefinitelyモードになっているか

3. **代替手段**
```bash
# ターミナルからスリープ防止
caffeinate -d -i -m -s &

# 学習実行
python3 janken_train_new_safe.py

# 完了後にcaffeinateを停止
killall caffeinate
```

## 📊 学習時間の最適化

### 効率的な学習スケジュール
```
🌅 朝の学習（推奨）
- 時間：7:00-8:00
- 設定：Caffeine 1時間モード
- 利点：新鮮な環境、十分なバッテリー

🌙 夜間学習（長時間可能）
- 時間：23:00-翌朝
- 設定：Caffeine Indefinitelyモード + 電源接続
- 利点：邪魔されない、長時間学習可能

⚡ 短時間集中学習
- 時間：昼間の空き時間
- 設定：Caffeine 30分モード
- 利点：手軽、確実に完了
```

## 🎯 学習効率向上のコツ

### 1. 事前準備の徹底
- データの準備完了確認
- 環境設定の事前テスト
- 必要な時間の計算

### 2. 監視とログ
```bash
# 学習ログの保存
python3 janken_train_new_safe.py 2>&1 | tee training_log.txt

# ログの確認
tail -f training_log.txt
```

### 3. 並行作業の回避
- 学習中は他の重い処理を避ける
- ブラウザのタブを最小限に
- 不要なアプリケーションを終了

## 📱 代替手段・バックアップ方法

### 1. ターミナルベースの解決
```bash
# caffeinate コマンド使用
caffeinate -d python3 janken_train_new_safe.py
```

### 2. システム設定による方法
```bash
# 一時的な設定変更
sudo pmset -c sleep 0
sudo pmset -c displaysleep 0

# 学習実行後に戻す
sudo pmset -c sleep 10
sudo pmset -c displaysleep 5
```

### 3. tmux/screenセッション
```bash
# tmuxセッション作成
tmux new -s ai_training

# Caffeine + 学習実行
python3 janken_train_new_safe.py

# セッションから離脱（Ctrl+B, D）
# 後で再接続: tmux attach -s ai_training
```

## 🎉 成功パターン

### 最も安全で確実な方法
1. **事前準備**（5分）
   - 電源接続
   - Caffeine起動・ON
   - データ確認

2. **学習実行**（20-40分）
   - `python3 janken_train_new_safe.py`
   - 進行状況を定期確認

3. **後処理**（2分）
   - Caffeine OFF
   - 結果確認
   - ファイル保存確認

### 成功率を上げるコツ
- **小分けにする**：長時間学習より短時間×複数回
- **確認を怠らない**：各段階で状況をチェック
- **予備時間を設ける**：余裕を持ったスケジュール

---

## 📞 サポート情報

### 公式情報
- **開発者**：IntelliScape
- **公式サイト**：GitHub - caffeine
- **バージョン確認**：アプリ > About Caffeine

### 代替アプリ
もしCaffeineで問題が発生した場合：
- **Amphetamine**（App Store）
- **KeepingYouAwake**（無料・オープンソース）
- **Lungo**（App Store）

---

**💡 重要なポイント**
Caffeineは学習の邪魔をなくすツールです。適切に使用することで、安心してじゃんけんAIの学習を行えます！

**🎯 目標達成への道**
Caffeine + 安全版スクリプト + 適切なデータ = 高精度じゃんけんAI ✨