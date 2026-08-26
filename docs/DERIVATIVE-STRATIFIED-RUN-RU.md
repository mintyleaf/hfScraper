# Запуск расчёта fine-tune и adapter моделей

Команда работает только с текстовыми LLM, созданными с 1 января по 31 декабря года, указанного в `created_from`/`created_to` конфигурации. В текущем профиле это 2025 год.

## Что именно делает проход

1. Читает готовый каталог Hugging Face из `results-2025-production/cache/catalog-2025.json.gz`.
2. Выбирает все текстовые репозитории, классифицированные метаданными как `finetune` или `adapter`.
3. Для full fine-tune использует размер полного checkpoint. Для adapter/LoRA/QLoRA использует размер объявленной базовой модели, а не размер файла адаптера.
4. Офлайн просматривает пять локальных Parquet-шардов. На этом шаге определяются наличие model card и опубликованный GPU time/cost.
5. Все репозитории с опубликованным compute включаются полностью. Остальные делятся на страты по типу, размеру модели и наличию карточки в Parquet. Модели от 34B включаются полностью; из остальных берётся детерминированная случайная выборка до 300 репозиториев на страту.
6. Только для manifest скачиваются отсутствующие README и файлы `adapter_config.json`, `trainer_state.json`, `training_args.json`. Ответы кэшируются; HTTP 429 повторяются с задержкой, повторный запуск продолжает работу по кэшу.
7. Регулярные выражения и локальная LLM извлекают только опубликованные параметры обучения с дословным evidence: total tokens, dataset rows, average tokens, epochs, steps, batch, gradient accumulation и world size. LLM не придумывает отсутствующие значения.
8. Число токенов определяется в таком порядке:
   - явное total training tokens;
   - `rows × average_tokens × epochs`;
   - `steps × per_device_batch × gradient_accumulation × world_size × average_tokens`.
9. Если опубликован GPU time/cost, используется он. Иначе считается `FLOPs = 6 × N_params × D_tokens`, затем H100 GPU-hours при 989 TFLOPS и 40% эффективности и стоимость по `$1.85/GPU-hour`.
10. Для нераскрытой части считается арифметическое среднее стоимости случайной выборки отдельно в каждой страте и умножается на число репозиториев этой страты. Медиана 549 добровольно раскрывших compute запусков для экстраполяции не используется.
11. Bootstrap по стратам даёт нижнюю и верхнюю границы sampling uncertainty. В отчёте отдельно показаны coverage, неоцениваемая часть и ограничения.

## Одна команда боевого запуска из корня проекта

Если Ollama работает на той же машине:

```bash
mkdir -p results-2025-derivatives-stratified && HF_LLM_BASE_URL=http://127.0.0.1:11434/v1 HF_LLM_MODEL=qwen2.5:14b go run ./cmd/hfscraper derivative-sample --config configs/catalog.2025-derivatives.stratified.json --execute 2>&1 | tee results-2025-derivatives-stratified/console.log
```

Если Ollama работает на другом компьютере, замените `127.0.0.1` на доступный с удалённой машины IP этого компьютера.

## Только план, без Hugging Face и без LLM

```bash
go run ./cmd/hfscraper derivative-sample --config configs/catalog.2025-derivatives.stratified.json
```

План печатает точные `manifest`, `targeted_repositories` и `maximum_http_requests`. Боевой режим всегда сначала строит тот же план, поэтому отдельный плановый запуск не обязателен.

## Оценка времени

Жёсткий верхний предел конфигурации — 12 000 репозиториев и до 48 000 HTTP-запросов. При двух HTTP workers, интервале 750 мс и без 429 это около 5 часов сетевого времени; с rate limit разумно закладывать 6–18 часов. Для 12 000 LLM-вызовов при четырёх workers и 5–20 секундах на карточку — примерно 4–17 часов. Часть стадий перекрывается слабо, поэтому безопасная оценка полного запуска — 10–36 часов. Точная оценка уточняется по числам plan-only и реальной скорости первых 100 LLM-вызовов.

Все загрузки и LLM-ответы кэшируются. После остановки запускается та же команда.

## Результаты

- `derivative-plan-summary.json` — точная population/coverage и объём работы;
- `derivative-sampling-manifest.json` и `.csv` — выбранные репозитории и веса;
- `derivative-evidence.json` — извлечённые параметры и evidence;
- `derivative-results.json` и `.csv` — расчёт по каждому выбранному репозиторию;
- `derivative-summary.json` — агрегированный итог;
- `DERIVATIVE-TRAINING-COST-2025-RU.md` — итоговый человекочитаемый отчёт.
