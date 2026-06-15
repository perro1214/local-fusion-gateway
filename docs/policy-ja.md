# OpenRouter FusionをローカルLLMで使うための方針

作成日: 2026-06-15
最終更新日: 2026-06-16

## 目的

OpenRouter Fusionの「複数モデルの並列回答をjudgeが比較し、最終回答に統合する」仕組みを、Ollama、LM Studio、llama.cppなどのローカルLLMでも使えるようにする。

本プロジェクトでは、OpenRouterのサーバー側実装を完全に再現するのではなく、OpenAI互換APIを持つローカルLLMを束ねるローカルGatewayを作る方針とする。既存のOpenAI SDK利用アプリからは、base URLをこのGatewayへ差し替えるだけで使える状態を目指す。

## 結論

v1は「OpenAI互換のローカルFusion Gateway」として実装を進める。2026-06-16時点で、最小Gateway、Fusion orchestration、Gemini API smoke test、judge構造化出力、観測用debug metadataまで実装済みである。

- 実装形態: Python + FastAPI + httpx + Pydantic
- 公開API: `POST /v1/chat/completions`
- 対象LLM: OpenAI互換APIを提供するローカル実行基盤
- 初期対象例: Ollama、LM Studio、llama.cpp server
- 互換目標: OpenRouter Fusionの最小API互換
- Fusion発火: v1では明示発火を優先
- Web検索/Fetch: v1では既定無効
- structured judge: `response_format: json_schema` を使い、非対応backendでは旧方式へfallback
- observability: `X-Request-ID`、構造化ログ、`X-Local-Fusion-Debug: true` による安全なmetadata表示

完全互換よりも、ローカルで確実に動くこと、データを外へ出さないこと、既存クライアントから使いやすいことを優先する。

## 現在の実装状況

実装済み機能は次の通り。

- `GET /health`
- `POST /v1/chat/completions`
- 設定済みOpenAI互換backendへの通常proxy
- `model: "openrouter/fusion"` によるFusion実行
- `tool_choice: "required"` と `tools: [{ "type": "openrouter:fusion" }]` によるFusion実行
- panel modelの並列実行
- panel一部失敗時のdegraded継続
- 全panel失敗時のhard failure
- judge modelによるFusion analysis JSON生成
- judge structured outputのための `response_format: json_schema`
- `response_format` 非対応backend向けのjudge retry
- judge JSON parse失敗時のpanel-only synthesis fallback
- `X-Request-ID` の生成/継承/レスポンス返却
- `X-Local-Fusion-Debug: true` 指定時の `local_fusion` metadata返却
- Gemini API OpenAI互換endpointを使ったsmoke test

直近の確認では、`models/gemini-2.5-flash-lite` を使い、Geminiのproxy経路とFusion経路の両方で `200` を確認済みである。debug metadataにはrequest id、panel/judge/synthesis latency、failed models、analysis有無、degraded reasonを含める。一方で、user prompt本文やpanel回答本文は含めない。

## OpenRouter Fusionの理解

OpenRouter Fusionは、単体モデルでは不十分なタスクに対して、複数モデルの回答を並列に集め、judge modelが差分を構造化して、outer modelが最終回答を書く仕組みである。

公式仕様上の主要な流れは次の通り。

1. クライアントが `model: "openrouter/fusion"`、または `tools: [{ "type": "openrouter:fusion" }]` を指定する。
2. outer modelがFusionを呼ぶか判断する。
3. panel model群が同じpromptに並列回答する。
4. judge modelがpanel回答を比較し、構造化JSONを作る。
5. outer modelがjudge分析を使って最終回答を書く。

judgeは回答を単純に混ぜるのではなく、次の観点で比較する。

- `consensus`: 多くのpanelが同意した点
- `contradictions`: panel間で矛盾した点
- `partial_coverage`: 一部のpanelだけが触れた点
- `unique_insights`: 特定panelだけの有用な洞察
- `blind_spots`: panel全体が見落とした可能性のある点

OpenRouter公式では、panelとjudgeの内部呼び出しで `openrouter:web_search` と `openrouter:web_fetch` が使える。ただし、ローカルGateway v1ではプライバシーと実装安定性を優先し、これらは既定では無効にする。

## v1で作るもの

v1は、OpenRouter Fusionの概念と主要APIだけをローカルで使えるようにする。

### Gateway

ローカルでFastAPIサーバーを起動し、OpenAI互換のChat Completions APIを公開する。

想定エンドポイント:

```text
POST http://localhost:8080/v1/chat/completions
```

既存クライアントは次のように差し替える。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="local-fusion",
)
```

### ローカルモデル接続

各ローカルLLMは、OpenAI互換APIのbase URLとmodel名で登録する。

```yaml
models:
  ollama-qwen:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    model: "qwen3:8b"

  lmstudio-llama:
    base_url: "http://localhost:1234/v1"
    api_key: "lm-studio"
    model: "local-model"
```

`analysis_models` では、外部プロバイダのslugではなく、この設定ファイル上の論理名を指定する。

### Fusion呼び出し

v1では次のどちらかをFusion実行として扱う。

```json
{
  "model": "openrouter/fusion",
  "messages": [
    { "role": "user", "content": "Compare ridge, lasso, and elastic-net regression." }
  ]
}
```

または:

```json
{
  "model": "ollama-qwen",
  "tool_choice": "required",
  "messages": [
    { "role": "user", "content": "Compare ridge, lasso, and elastic-net regression." }
  ],
  "tools": [
    {
      "type": "openrouter:fusion",
      "parameters": {
        "analysis_models": ["ollama-qwen", "lmstudio-llama"],
        "model": "ollama-qwen"
      }
    }
  ]
}
```

OpenRouter本家のように「outer modelが必要時だけFusionを呼ぶ」挙動はv2以降に回す。ローカルLLMはtool calling品質がモデルやテンプレートに強く依存するため、v1で自動判断にすると挙動が不安定になりやすい。

## API互換範囲

v1で受け付けるFusion関連パラメータは次の通り。

| パラメータ | 方針 |
| --- | --- |
| `analysis_models` | panelとして並列実行するローカルモデル論理名。1から8件。 |
| `model` | judge model。省略時は先頭のpanel model、またはGateway既定judge。 |
| `max_tool_calls` | v1では受け付けるが、Web tool無効時は実質未使用。将来互換のため保持。 |
| `max_completion_tokens` | panel/judge/最終合成の出力上限として転送する。 |
| `reasoning` | 対応するローカル実行基盤にだけ転送する。未対応なら無視してログに残す。 |
| `temperature` | panel/judge呼び出しへ転送する。 |
| `tools` | v1では空配列または未指定を推奨。Web toolは実行しない。 |

通常のOpenAI互換Chat Completionsパラメータは、可能な範囲で下流のローカルLLMへ転送する。ただし、ローカル実行基盤ごとに未対応パラメータがあるため、未対応項目はエラーにせず、ログに記録して無視する方針とする。

## 実行フロー

Fusion対象リクエストの処理は次の順序にする。

1. リクエストをPydanticで検証する。
2. Fusion設定を解決する。
3. `analysis_models` の各モデルへ同じ `messages` を並列送信する。
4. 成功したpanel回答を集め、失敗したモデルを `failed_models` に記録する。
5. 成功回答が1件もなければhard failureを返す。
6. judge modelへpanel回答を渡し、構造化JSONを生成させる。
7. judge JSONを検証する。
8. judgeが成功した場合はanalysisを使って最終回答を生成する。
9. judgeが失敗した場合はdegraded modeとして、panel回答だけを使って最終回答を生成する。
10. OpenAI互換のChat Completionsレスポンスとして返す。

## judge出力スキーマ

judgeには次のJSONを要求する。

```json
{
  "consensus": ["string"],
  "contradictions": [
    {
      "topic": "string",
      "stances": [
        { "model": "string", "stance": "string" }
      ]
    }
  ],
  "partial_coverage": [
    {
      "models": ["string"],
      "point": "string"
    }
  ],
  "unique_insights": [
    {
      "model": "string",
      "insight": "string"
    }
  ],
  "blind_spots": ["string"]
}
```

ローカルLLMはJSON遵守が不安定な場合があるため、v1では次の順に扱う。

1. そのままJSON parseを試す。
2. markdown code block内のJSON抽出を試す。
3. 失敗したらjudge degradationとしてanalysisなしで続行する。

現行実装では、judge呼び出しにOpenAI互換の `response_format` を付ける。

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "fusion_analysis",
      "schema": {
        "type": "object",
        "required": [
          "consensus",
          "contradictions",
          "partial_coverage",
          "unique_insights",
          "blind_spots"
        ]
      }
    }
  }
}
```

`response_format` を受け付けないbackendでは、judge呼び出しが `400` または `422` になった場合だけ、`response_format` なしで一度retryする。このretryも失敗する、またはjudge出力がschemaに合わない場合は、従来通りanalysisなしのdegraded modeへ落とす。

## 失敗時の扱い

OpenRouter Fusionの考え方に合わせ、部分失敗ではなるべく処理を継続する。

### panelの一部失敗

少なくとも1つのpanelが成功した場合は成功扱いにする。

### judge失敗

panelが成功してjudgeだけ失敗した場合は、analysisを省略してdegraded modeで最終回答を作る。

### hard failure

全panelが失敗した場合のみhard failureとする。

想定する `failure_reason`:

- `all_panels_failed`
- `rate_limited`
- `insufficient_credits`
- `judge_not_valid_json`
- `judge_schema_mismatch`
- `judge_upstream_error`
- `judge_empty_completion`
- `unexpected_error`

ローカルLLMでは `insufficient_credits` は通常発生しないが、OpenRouter互換のため予約語として残す。

## Web検索とWeb Fetch

v1では `openrouter:web_search` と `openrouter:web_fetch` は既定無効とする。

理由:

- ローカルLLM利用の主目的は、入力を外部サービスへ出さないことにある。
- 検索APIキー、引用、ドメイン制御、prompt injection対策まで含めるとv1の範囲が大きくなりすぎる。
- OpenRouter公式のserver toolsはbetaであり、互換対象として固定しにくい。

ただし、将来追加できるように `tools` フィールドは受け付ける。v1では `tools: []` または未指定を推奨し、Web toolが指定された場合は「未対応」としてログに残す。

## セキュリティとプライバシー

ローカルGatewayは、OpenRouterよりも機能を減らす代わりに、データ境界を明確にする。

- 既定では外部ネットワークへpromptを送らない。
- ローカルLLMのbase URLは明示設定されたものだけ許可する。
- 任意URLへのproxy機能はv1では持たせない。
- request/responseログは既定で本文を保存しない。
- debug modeで本文ログを有効化する場合は、明示的な設定を必要にする。
- API keyはローカル用途では固定値でもよいが、LAN公開する場合は必ず認証を有効にする。

LANや社内ネットワークで使う場合は、次を推奨する。

- `127.0.0.1` bindを既定にする。
- 外部公開時だけ `0.0.0.0` bindを許可する。
- CORSは既定拒否にする。
- 接続先base URLをallowlist化する。

## Observability

Fusionは通常の単体推論より遅く、重い。v1から最低限の計測を入れる。

記録する項目:

- request id
- Fusion実行有無
- panel model名
- judge model名
- panelごとの成功/失敗
- panelごとのlatency
- judge latency
- final synthesis latency
- input/output token数が取れる場合はその値
- degraded mode発生有無
- failure reason

OpenRouterの課金情報に相当するものはローカルでは取得できないため、v1では時間とtoken使用量を中心に見る。

現行実装では次の形で観測性を入れている。

- `X-Request-ID`: 指定があれば継承し、なければGatewayが生成する。
- 構造化ログ: proxy/Fusionの成功失敗、latency、degraded reasonをログに出す。
- debug metadata: `X-Local-Fusion-Debug: true` のときだけ、通常レスポンスに `local_fusion` を追加する。

`local_fusion` の内容は次の範囲に限定する。

- `request_id`
- `panel_models`
- `judge_model`
- `failed_models`
- `analysis_present`
- `degraded_reason`
- `latency_ms.total`
- `latency_ms.panels`
- `latency_ms.judge`
- `latency_ms.synthesis`

debug metadataには、通常の安全側の方針として、user prompt本文、panel回答本文、judge raw JSON、API keyは入れない。

## 実装フェーズ

### Phase 1: 方針確定と最小Gateway

- `POST /v1/chat/completions` をFastAPIで実装する。
- 非Fusionリクエストは指定されたローカルモデルへ単純proxyする。
- `model: "openrouter/fusion"` を検出できるようにする。
- 設定ファイルからローカルモデル定義を読み込む。

### Phase 2: panel並列実行

- `analysis_models` を解決する。
- `httpx.AsyncClient` でpanelを並列呼び出しする。
- timeout、接続失敗、HTTP errorをモデル単位で記録する。
- 1件以上成功したら処理継続する。

### Phase 3: judgeとanalysis

- panel回答をjudge promptへ整形する。
- judgeに構造化JSONを要求する。
- JSON parseと軽いcode block抽出を行う。
- schema不一致時はdegraded modeへ落とす。

### Phase 4: final synthesis

- analysisありの場合は、consensus、contradictions、blind spotsを明示して最終回答を生成する。
- analysisなしの場合は、panel回答一覧だけから最終回答を生成する。
- OpenAI互換Chat Completionsレスポンスに整形する。

### Phase 5: 検証と運用最低限

- 単体テストを追加する。
- OllamaまたはLM Studioを使った手動検証手順をREADMEに書く。
- ログとrequest idを整える。
- 失敗時の挙動を固定する。

### Phase 6: structured judgeと観測性

- judge呼び出しに `response_format: json_schema` を追加する。
- `response_format` 非対応backendでは旧方式へretryする。
- `X-Request-ID` を導入する。
- `X-Local-Fusion-Debug: true` で安全なmetadataを返す。
- Gemini APIのOpenAI互換endpointでsmoke testを行う。

Phase 6までは実装済みである。次の候補は、実サーバーとしてのcurl/OpenAI SDK検証、Ollama/LM Studio実機検証、またはdebug metadataの詳細化である。

## v1でやらないこと

- OpenRouterのserver tool実行基盤の完全再現
- `openrouter:web_search` の実行
- `openrouter:web_fetch` の実行
- 外側モデルによるFusion自動発火判断
- provider routing、fallback provider選択、課金管理
- OpenRouter generation metadata完全互換
- streamingの完全対応
- multimodal入力のFusion

streamingは重要だが、panel並列、judge、final synthesisと段階が多く、v1で入れると設計が複雑になる。まず非streamingで正しい挙動を作り、v2以降でイベントストリーム化する。

## テスト方針

最低限、次のケースを自動テストまたは手動検証する。

- 非Fusionリクエストが単一ローカルモデルへproxyされる。
- `model: "openrouter/fusion"` でFusionが実行される。
- `tool_choice: "required"` と `openrouter:fusion` でFusionが実行される。
- `analysis_models` が1件でも動く。
- `analysis_models` が2件以上のとき並列実行される。
- panelの一部失敗でdegraded成功する。
- 全panel失敗でhard failureになる。
- judgeが不正JSONを返したとき、analysisなしで最終回答が生成される。
- 未対応パラメータがあってもGatewayが落ちない。
- Web tool指定時、v1では実行せずログに残る。
- judgeが `response_format` に対応する場合、schema準拠analysisが使われる。
- judgeが `response_format` 非対応の場合、旧方式へretryされる。
- `X-Request-ID` がレスポンスheaderに返る。
- debug header指定時だけ `local_fusion` metadataが返る。
- debug metadataにprompt本文やpanel回答本文が含まれない。

## 参考資料

- OpenRouter Fusion Server Tool: https://openrouter.ai/docs/guides/features/server-tools/fusion
- OpenRouter Fusion Router: https://openrouter.ai/docs/guides/routing/routers/fusion-router
- OpenRouter Server Tools: https://openrouter.ai/docs/guides/features/server-tools/overview
- OpenRouter API Reference: https://openrouter.ai/docs/api/reference/overview
- Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- LM Studio OpenAI compatibility: https://lmstudio.ai/docs/developer/openai-compat
- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Gemini structured outputs: https://ai.google.dev/gemini-api/docs/structured-output

## 重要な前提

OpenRouterのFusionとserver toolsはbetaであり、APIや挙動が変わる可能性がある。このプロジェクトでは、公式仕様への追従よりも、ローカルLLMで安定して使える最小合議システムを優先する。

したがって、v1の成功条件は「OpenRouter Fusionと完全に同じレスポンスを返すこと」ではない。成功条件は、既存のOpenAI互換クライアントからローカルGatewayを呼び、複数のローカルLLMの回答をpanel + judge + synthesisで統合できることである。
