# Конфигурируемый каталог Hugging Face

Режим `catalog` строит несколько выборок моделей за один проход Hugging Face API,
определяет владельцев репозиториев и сохраняет результаты в CSV и JSON.

## Запуск

```bash
go run ./cmd/hfscraper catalog --config configs/catalog.example.json
```

Расширенный набор сценариев запускается так:

```bash
go run ./cmd/hfscraper catalog --config configs/catalog.scenarios.json
```

Другой каталог результатов задаётся без редактирования JSON:

```bash
go run ./cmd/hfscraper catalog --config configs/my-config.json --output my-results
```

Для gated public repositories можно передать read-only токен через `HF_TOKEN`.

## Готовые выборки

В `configs/catalog.example.json` уже настроены:

- 100 самых скачиваемых новых LLM за 365 дней;
- 100 самых скачиваемых base LLM;
- 100 самых скачиваемых adapters и LoRA;
- все LLM от 64B до 128B;
- 100 самых скачиваемых LLM от 7B до 14B.

Выборки независимы. Их можно удалять, копировать и менять. Самая ранняя дата
среди выборок определяет глубину единственного прохода API.

## Выходные файлы

В `output_dir` создаются:

- `models.csv` и `models.json` — модели, выборка, тип, параметры, популярность и
  compute estimates;
- `owners.csv` и `owners.json` — уникальные пользователи и организации;
- `summary.json` — числа моделей и владельцев, типы, downloads и likes по каждой
  выборке.
- `catalog.log.jsonl` — структурированные события, повторы HTTP и ошибки.

Одна модель может входить в несколько выборок. В статистике владельца один и тот
же repo учитывается один раз.

## Сканирование

Секция `scan` управляет общим проходом API.

- `now` фиксирует момент расчёта. Пустое значение использует текущее UTC-время.
- `page_size` задаёт от 1 до 1000 моделей на API-страницу.
- `max_pages` ограничивает проход; ноль сканирует до самой ранней даты выборок.
- `require_weights` исключает репозитории без распознанных весов.
- `resolve_base_parameters` получает размер базовой модели для adapters в
  выборках с диапазоном параметров или compute-профилем.
- `timeout_seconds`, `retries` и `quiet` управляют HTTP и прогрессом.

Если проход ограничен `max_pages`, `summary.json` содержит `complete: false`.

## Логирование и ошибки

Секция `logging` задаёт `level` (`debug`, `info`, `warn`, `error`), имя JSONL-файла
`file`, дублирование в `stderr`, подробные `http_requests` и логирование обычных
`skipped_records`. HTTP-повторы и окончательные ошибки логируются независимо от
детализации успешных запросов.

Ошибки страницы каталога и записи результатов фатальны. Ошибка одного owner или
base-model enrichment не уничтожает всю выборку: причина сохраняется в owner
`error` или model `errors`, а также в JSONL-логе.

## Фильтры выборки

Каждый объект `selections` имеет уникальное `name` и описание `description`.

### Период

Можно указать `lookback_days` либо точные `created_from` и `created_to` в формате
`YYYY-MM-DD` или RFC3339. По умолчанию берутся последние 365 дней.

### Тип модели

`model_kinds` принимает любое сочетание:

- `base` — нет признаков производной модели;
- `finetune` — полное дообучение;
- `adapter` — PEFT, LoRA, QLoRA и adapters;
- `quantized` — квантованная публикация;
- `merge` — merge нескольких моделей.

Пустой список не фильтрует тип. `base` является эвристикой: Hugging Face не имеет
обязательного флага, доказывающего обучение с нуля.

### Размер модели

- `min_parameters_b` и `max_parameters_b` задают включительный диапазон в
  миллиардах параметров;
- `include_unknown_parameters` оставляет модели без известного размера.

Для adapter по возможности используется размер `base_model`. Если связь с базой
не указана или её метаданные недоступны, остаётся собственный parameter count
adapter-файла.

### LLM, библиотеки и имена

- `pipeline_tags` фильтрует, например, `text-generation` и
  `text2text-generation`;
- `libraries` фильтрует `transformers`, `peft`, `diffusers` и другие библиотеки;
- `model_id_regex` применяет Go regular expression к полному repo ID.

### Теги

- `tags_any` требует хотя бы один тег;
- `tags_all` требует все перечисленные теги;
- `exclude_tags` исключает при совпадении любого тега.

Сравнение тегов регистронезависимое и точное.

### Популярность

Доступны `min_downloads`, `max_downloads`, `min_likes` и `max_likes`. Ноль
отключает соответствующую границу.

### Сортировка и лимит

`sort_by` принимает `downloads`, `likes`, `created_at`, `parameters` или
`repo_id`. `sort_direction` принимает `asc` либо `desc`.

`limit` применяется после фильтрации и сортировки. Ноль сохраняет все модели.

### Тип владельца

`owner_types` может содержать `user`, `organization` и `unknown`. Фильтр
применяется после получения профилей. Пустой список оставляет все типы.

## Владельцы

Секция `owners` включает обогащение профилей и задаёт число `workers`.

Сохраняются публичные поля: тип, URL профиля, полное имя, verification/pro/plan,
followers, число моделей, datasets, Spaces, papers, members организации, дата
создания и описание. Добавляется статистика отобранных репозиториев владельца.

## Compute-профили

`compute_profiles` описывает произвольные кластеры:

- `gpu_name` — подпись GPU;
- `gpu_tflops` — производительность одного GPU;
- `efficiency` — фактическая доля пиковой производительности от 0 до 1;
- `gpu_hour_cost_usd` — цена GPU-часа;
- `tokens_per_parameter` — обучающие токены на параметр;
- `machines` — число машин;
- `gpus_per_machine` — GPU в машине;
- `finetune_compute_fraction` — доля полного обучения для fine-tune и adapter.

Выборка подключает профили своим списком `compute_profiles`. Для каждой модели
рассчитываются GPU-hours, общее число GPU, wall-clock days, стоимость и
использованный training fraction.

Пример содержит 1000 машин с одним H100 и 256 машин с четырьмя H100. Это 1000 и
1024 GPU соответственно, поэтому ожидаемая длительность почти одинакова. Можно
подставить характеристики любого consumer GPU.

```text
tokens = tokens_per_parameter × parameters
FLOPs = 6 × parameters × tokens × training_fraction
GPU_hours = FLOPs / (gpu_tflops × 10¹² × 3600 × efficiency)
wall_days = GPU_hours / (machines × gpus_per_machine × 24)
cost = GPU_hours × gpu_hour_cost_usd
```

Оценка не моделирует сеть, память, checkpointing, отказы машин и снижение
эффективности при масштабировании.
