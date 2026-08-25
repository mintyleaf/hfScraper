# Стратифицированная оценка рынка 2025: запуск и методика

## Что подготовлено

Новый режим `market-sample` оценивает стоимость base-обучений без попытки скачать README каждого репозитория Hugging Face.

Он использует:

- готовый API-checkpoint на 901 931 репозиторий с весами;
- пять уже скачанных Parquet-шардов;
- полный census потенциально дорогих base-моделей от 34B параметров;
- стратифицированную выборку остальных base-кандидатов;
- точечное скачивание только отсутствующих в Parquet README;
- локальную OpenAI-compatible LLM;
- отдельные формулы для text и diffusion;
- frozen baseline завершённого no-LLM запуска для подтверждённого lower bound и опубликованного derivative compute.

Режим не использует Codex, OpenAI API или токены этой переписки. Все LLM-вызовы идут только на указанный локальный endpoint.

## Уже построенный plan

Plan-only проход выполнен локально, без HTTP-запросов и LLM:

| Показатель | Значение |
|---|---:|
| Base-кандидатов | 103 172 |
| Text base-кандидатов | 94 146 |
| Diffusion base-кандидатов | 9 026 |
| Есть полная карточка в Parquet | 21 614 |
| Нет карточки в Parquet | 81 558 |
| Manifest для проверки | 3 327 |
| Полный census >=34B | 610 |
| Карточки manifest уже локальны | 1 488 |
| Требуется targeted README | 1 839 |

По рынкам:

| Рынок | Manifest | Census >=34B | Targeted README |
|---|---:|---:|---:|
| Text LLM | 2 107 | 607 | 1 183 |
| Diffusion | 1 220 | 3 | 656 |

Manifest детерминирован seed `2025`: повторный plan-only запуск создаёт ту же выборку при неизменном checkpoint и Parquet.

## Как построена выборка

Сначала берутся только предварительные API base-кандидаты. Явные fine-tune, adapter, fork, merge и quantized не участвуют в статистической base-экстраполяции.

Base-кандидаты разделяются одновременно по трём признакам:

1. Рынок: `text_llm` или `diffusion`.
2. Диапазон параметров: `<2B`, `2–<7B`, `7–<13B`, `13–<34B`, `34–<70B`, `70–<120B`, `120–<500B`, `>=500B`, `unknown`.
3. Источник карточки: полная карточка есть в Parquet или отсутствует.

Все известные модели от 34B проверяются полностью и имеют sampling weight `1`. Это сделано потому, что небольшое количество крупных моделей формирует непропорционально большую часть расходов.

Для каждого слоя ниже 34B и слоя с неизвестными параметрами выбирается до 150 моделей. Порядок определяется SHA-256 от `seed + repo_id`, поэтому выборку можно воспроизвести без зависимости от порядка map или файлов.

Каждая sampled-строка получает:

```text
inclusion_probability = sample_size / stratum_population
sampling_weight       = stratum_population / sample_size
```

Оценка слоя — сумма стоимости sampled-моделей, умноженной на их sampling weight. Census-модели не экстраполируются.

## Что делает plan-only

Команда:

```bash
go run ./cmd/hfscraper market-sample \
  --config configs/catalog.2025-stratified-market.local-llm.json
```

Она:

1. Загружает локальный catalog checkpoint.
2. Выделяет text/diffusion base-кандидатов.
3. Читает пять локальных Parquet-шардов.
4. Измеряет точное покрытие карточек.
5. Создаёт manifest и sampling weights.
6. Завершается до сети, README, LLM и расчёта стоимости.

Результаты plan-only:

- `results-2025-stratified-market/sampling-plan-summary.json`;
- `results-2025-stratified-market/sampling-manifest.json`;
- `results-2025-stratified-market/sampling-manifest.csv`;
- `results-2025-stratified-market/sampling.log.jsonl`.

Перед боевым запуском можно проверить число будущих запросов:

```bash
jq '{base_candidates, parquet_covered_candidates, parquet_missing_candidates, manifest_entries, census_entries, targeted_readme_requests}' \
  results-2025-stratified-market/sampling-plan-summary.json
```

Ожидаемое `targeted_readme_requests`: **1 839**, не 244 тысячи.

## Подготовка локальной LLM

Рекомендуемый минимум — Qwen2.5 14B или другая instruction-модель, которая стабильно возвращает JSON. Если памяти недостаточно, можно использовать 7B, но качество lineage-классификации будет ниже.

Пример с Ollama:

```bash
ollama pull qwen2.5:14b
ollama serve
```

Если Ollama уже запущена как system service, второй процесс запускать не нужно.

Проверка endpoint:

```bash
curl -s http://127.0.0.1:11434/v1/models
```

## Боевой запуск

Только флаг `--execute` разрешает сетевые README-запросы и LLM:

```bash
HF_LLM_BASE_URL=http://127.0.0.1:11434/v1 \
HF_LLM_MODEL=qwen2.5:14b \
go run ./cmd/hfscraper market-sample \
  --config configs/catalog.2025-stratified-market.local-llm.json \
  --execute
```

HF-токен не обязателен. Настроен один README worker, пауза 500 ms, восемь retry и соблюдение `Retry-After`/`RateLimit` при HTTP 429.

При текущем наблюдаемом ограничении примерно 12 README за пять минут только targeted-fetch может занять около 13 часов. Локальная LLM затем обработает 3 327 карточек; время зависит от железа и модели. Реалистичный общий порядок — 14–24 часа.

## Resume после остановки

Можно нажать `Ctrl+C`, затем повторить ту же команду.

Уже полученные README сохраняются по hash репозитория в:

```text
results-2025-stratified-market/cache/targeted-readmes/
```

Недоступные README получают `.missing` marker и не запрашиваются повторно. Успешные LLM-ответы дописываются с `fsync` в:

```text
results-2025-stratified-market/cache/stratified-local-llm-review-v1.jsonl
```

Cache key включает prompt version, scope text/diffusion, card hash и metadata. При неизменной модели и prompt повторный запуск продолжает работу, а не начинает сначала.

## Что возвращает локальная LLM

Для текущего репозитория выбирается один класс:

- `independent_base`;
- `continued_pretraining`;
- `finetune`;
- `adapter`;
- `fork_or_mirror`;
- `quantized`;
- `merge`;
- `non_text`;
- `unknown`.

Также возвращаются confidence, upstream, canonical training run, год обучения, total parameter count и короткие дословные evidence-фрагменты.

High/medium evidence и parameter evidence принимаются только если являются реальными подстроками model card. Год не принимается из свободного ответа LLM: он повторно извлекается только из дословной evidence.

Разные размеры одной base-семьи не объединяются. Одинаковый `canonical_training_run` с одинаковым числом параметров учитывается один раз.

## Как формируются три суммы

### Confirmed lower

Берётся frozen baseline завершённого no-LLM прохода:

```text
reports/baseline-2025-no-llm-summary.json
```

Он содержит подтверждённые $55.22 млн. Это нижняя граница, а не статистическая экстраполяция.

### Central

В base-оценку входят:

- deterministic independent-pretraining evidence;
- high-confidence `independent_base` локальной LLM;
- sampling weights для слоёв ниже 34B;
- все подтверждённые модели >=34B с весом `1`.

К base central добавляется только опубликованный compute fine-tune/adapter из baseline. Подтверждённая base-сумма baseline второй раз не добавляется.

### Upper

К central base добавляются medium-confidence `independent_base`. Производные модели без опубликованного compute по исходному условию не экстраполируются.

Границы являются evidence-сценариями, а не формальным 95% confidence interval. Систематическая проблема плохих model cards остаётся и указывается в отчёте.

## Параметры и формулы

Если API сообщил safetensors total parameters, используется это число. Если API-параметры неизвестны, локальная LLM может извлечь total parameter count только вместе с дословной parameter evidence из карточки. Без параметров модель классифицируется, но не получает выдуманную стоимость и попадает в `uncostable_unknown_parameters`.

Text:

```text
T     = reported pretraining tokens, иначе 20 × N_total
FLOPs = 6 × N_effective × T
cost  = FLOPs / (989e12 × 3600 × 0.4) × $1.85
```

Diffusion:

```text
FLOPs = 6 × N_effective × (N_total × 3 × 1024)
cost  = FLOPs / (989e12 × 3600 × 0.4) × $1.85
```

Diffusion budget остаётся отдельным явно указанным предположением.

## Итоговые файлы

После `--execute`:

- `sampling-results.json/csv` — каждая sampled/census модель, LLM evidence, sampling weight, raw и weighted cost;
- `sampling-summary.json` — итоги по рынкам и стратам;
- `sampling-report.md` — читаемый lower/central/upper;
- `sampling.log.jsonl` — progress, 429/retry и ошибки;
- `cache/targeted-readmes/` — resume-cache README;
- `cache/stratified-local-llm-review-v1.jsonl` — resume-cache LLM.

## Что этот метод всё ещё не обещает

- Это рыночная оценка публично видимых репозиториев, не бухгалтерия компаний.
- Приватные/удалённые модели и закрытые кластеры отсутствуют.
- Fine-tune/adapter без опубликованного compute остаются непосчитанными по исходному правилу.
- Unknown parameters не превращаются в произвольный средний размер.
- Результат зависит от качества стратификации и локального классификатора, поэтому evidence и каждая sampled-строка сохраняются для аудита.
