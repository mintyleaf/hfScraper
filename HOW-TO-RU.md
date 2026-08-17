# Catalog mode: полный reference

Документ описывает режим выборок моделей и владельцев. Старый режим подсчёта
scratch-моделей рассмотрен отдельно в конце.

## 1. Запуск

```bash
go run . catalog [flags]
```

Доступные CLI-флаги:

- `--config <path>` — JSON-конфиг. Default: `catalog.example.json`.
- `--output <path>` — переопределяет `output_dir` из JSON.
- `--help`, `-h` — справка.

Примеры:

```bash
go run . catalog
go run . catalog --config configs/llm.json --output results/llm
HF_TOKEN=hf_... go run . catalog --config catalog.example.json
```

`HF_TOKEN` используется для аутентифицированных запросов и доступных токену
gated repositories. В результаты токен не записывается.

## 2. Корневой объект конфига

Поддерживаются только следующие поля:

- `endpoint` — Hub URL. Default: `https://huggingface.co`.
- `output_dir` — директория результатов. Default: `catalog-results`.
- `logging` — уровни и место записи диагностики.
- `scan` — настройки общего прохода API.
- `owners` — настройки профилей владельцев.
- `compute_profiles` — массив GPU-кластеров.
- `selections` — массив независимых выборок; минимум одна.

Неизвестное поле вызывает ошибку, поэтому опечатки не игнорируются.

Минимальный конфиг:

```json
{
  "selections": [
    {
      "name": "latest_models",
      "lookback_days": 365
    }
  ]
}
```

## 3. Все опции scan

```json
"scan": {
  "now": "2026-08-16T00:00:00Z",
  "page_size": 1000,
  "max_pages": 0,
  "require_weights": true,
  "resolve_base_parameters": true,
  "timeout_seconds": 30,
  "retries": 5,
  "quiet": false
}
```

### scan.now

String в формате RFC3339 или `YYYY-MM-DD`. Пустая строка использует текущее
UTC-время. Фиксирует верхнюю границу и делает временное окно воспроизводимым.
Дата без времени означает `00:00:00 UTC`.

Default: текущее UTC-время.

### scan.page_size

Integer 1–1000. Число моделей в одной API-странице. Для полного прохода обычно
нужно 1000.

Default: 1000.

### scan.max_pages

Неотрицательный integer.

- `0` — сканировать до самой ранней даты среди selections.
- `N > 0` — остановиться после N страниц.

Остановка по лимиту обычно даёт `complete: false`.

Default: 0.

### scan.require_weights

Boolean. При `true` репозиторий должен иметь `safetensors.total` или распознанный
файл весов. Поддерживаются Safetensors, PyTorch, TensorFlow/Keras, Flax/JAX,
ONNX, TFLite, Paddle, MXNet, Core ML и несколько экспортных форматов.
Токенизаторы и trainer state весами не считаются.

Default: true.

### scan.resolve_base_parameters

Boolean. Получает `safetensors.total` базовой модели для adapter-кандидатов,
которые участвуют в selections с диапазоном параметров или compute-профилем.
Благодаря этому LoRA для 70B базы фильтруется и оценивается как 70B, а не по
размеру adapter.

Default: true.

### scan.timeout_seconds

Integer больше нуля. Timeout одного HTTP-запроса.

Default: 30.

### scan.retries

Integer от 0 до 20. Число повторов после временной ошибки. `0` отключает
повторы. При HTTP 429 клиент учитывает `Retry-After` и Hugging Face `RateLimit`.

Default: 5.

### scan.quiet

Boolean. `true` отключает промежуточный прогресс.

Default: false.

## 3A. Все опции logging

```json
"logging": {
  "level": "info",
  "file": "catalog.log.jsonl",
  "stderr": true,
  "http_requests": false,
  "skipped_records": false
}
```

### logging.level

Минимальный сохраняемый уровень: `debug`, `info`, `warn` или `error`.

Default: `info`.

### logging.file

JSONL-файл лога. Относительный путь считается от `output_dir`, абсолютный
используется как есть. Каждая строка является отдельным JSON-объектом с полями
`time`, `level`, `event`, `message` и контекстом события.

Default: `catalog.log.jsonl`.

### logging.stderr

При `true` события также печатаются в stderr в человекочитаемом виде. JSONL-файл
создаётся независимо от этого значения.

Default: true.

### logging.http_requests

При `true` добавляет debug-события каждого HTTP-запроса: URL, попытку, статус,
размер ответа и длительность. Повторы и окончательные HTTP-ошибки логируются и
при `false`. Для успешных запросов также требуется `level: "debug"`.

Default: false.

### logging.skipped_records

При `true` добавляет debug-события для обычных исключений: модель старше окна
или без распознанных весов. Некорректные даты и даты из будущего всегда являются
`warn`. Для обычных исключений также требуется `level: "debug"`.

Default: false.

## 4. Все опции owners

```json
"owners": {
  "enabled": true,
  "workers": 8
}
```

### owners.enabled

Boolean. Включает публичные overview-запросы. Сначала проверяется organization,
затем user. Если оба endpoint недоступны, тип равен `unknown`.

Default: true.

### owners.workers

Integer 1–64. Параллельность owner overview и base-parameter запросов.

Default: 8.

Если enrichment выключен, все `owner_type` будут `unknown`. Поэтому
`owner_types: ["user"]` или `["organization"]` удалит все результаты.

## 5. Все опции compute_profiles

```json
{
  "name": "h100_256x4",
  "description": "256 машин по четыре H100",
  "gpu_name": "NVIDIA H100",
  "gpu_tflops": 989,
  "efficiency": 0.4,
  "gpu_hour_cost_usd": 1.85,
  "tokens_per_parameter": 20,
  "machines": 256,
  "gpus_per_machine": 4,
  "finetune_compute_fraction": 0.05
}
```

### name

Обязательный уникальный string. Selection ссылается на профиль по этому имени.

### description

Необязательное описание профиля. На формулу не влияет.

### gpu_name

String только для подписи результата. На формулу не влияет.

### gpu_tflops

Number больше нуля. Пиковые TFLOPS одного GPU для используемого типа вычислений.

### efficiency

Number в диапазоне `(0, 1]`. Реально используемая доля пиковой
производительности; `0.4` означает 40%.

### gpu_hour_cost_usd

Number. Цена одного GPU-часа. Может быть 0, если нужна только длительность.

### tokens_per_parameter

Number больше нуля. Обучающие токены на параметр. В TAM/SAM формуле используется
20.

### machines

Integer больше нуля. Число машин.

### gpus_per_machine

Integer больше нуля. GPU в одной машине. Total GPUs равно
`machines × gpus_per_machine`.

### finetune_compute_fraction

Number от 0 до 1. Compute-множитель для `finetune` и `adapter`:

- `0.01` — 1% полного обучения;
- `0.05` — 5%;
- `0.10` — 10%;
- `1.0` — полный pretraining compute.

Для остальных model kinds fraction равен 1.

## 6. Compute-формула

```text
tokens = tokens_per_parameter × parameters
FLOPs = 6 × parameters × tokens × training_fraction
GPU_hours = FLOPs / (gpu_tflops × 10¹² × 3600 × efficiency)
total_gpus = machines × gpus_per_machine
wall_days = GPU_hours / total_gpus / 24
cost_usd = GPU_hours × gpu_hour_cost_usd
```

`wall_days` предполагает идеальное масштабирование и не моделирует сеть, память,
parallelism, checkpointing, простои и отказы.

## 7. Все опции selection

Полный объект:

```json
{
  "name": "llm_64_128b",
  "description": "LLM 64–128B за год",
  "lookback_days": 365,
  "created_from": "",
  "created_to": "",
  "pipeline_tags": ["text-generation", "text2text-generation"],
  "libraries": ["transformers", "peft"],
  "model_kinds": ["base", "finetune", "adapter"],
  "owner_types": ["user", "organization"],
  "tags_any": [],
  "tags_all": [],
  "exclude_tags": ["gguf", "gptq", "awq", "model-merge"],
  "model_id_regex": "(?i)qwen|llama",
  "min_parameters_b": 64,
  "max_parameters_b": 128,
  "include_unknown_parameters": false,
  "min_downloads": 0,
  "max_downloads": 0,
  "min_likes": 0,
  "max_likes": 0,
  "sort_by": "downloads",
  "sort_direction": "desc",
  "limit": 100,
  "compute_profiles": ["h100_1000x1", "h100_256x4"]
}
```

### name

Обязательный уникальный string. Записывается в `selection` и `summary.json`.

### description

Необязательный string. Сохраняется в JSON и summary, на фильтрацию не влияет.

### lookback_days

Неотрицательный integer. Используется, только если `created_from` пуст.
Отсчитывается от `created_to`, либо от `scan.now`, если `created_to` пуст.

Default: 365.

### created_from

String RFC3339 или `YYYY-MM-DD`. Нижняя включительная граница `createdAt`.
Если задана, `lookback_days` игнорируется.

### created_to

String RFC3339 или `YYYY-MM-DD`. Верхняя включительная граница. Пустое значение
использует `scan.now`. Дата без времени включает весь UTC-день.

### pipeline_tags

Array of strings. OR-логика, точное регистронезависимое сравнение. Пустой массив
отключает фильтр. Для LLM обычно:

```json
["text-generation", "text2text-generation"]
```

### libraries

Array of strings. OR-логика по `library_name`, например `transformers`, `peft`,
`diffusers`. Пустой массив отключает фильтр.

### model_kinds

Array с любым сочетанием `base`, `finetune`, `adapter`, `quantized`, `merge`.
OR-логика. Пустой массив разрешает все типы.

### owner_types

Array с любым сочетанием `user`, `organization`, `unknown`. Фильтр применяется
после owner enrichment, но перед финальным `limit`. Пустой массив разрешает всех.

### tags_any

Array of strings. Модель должна иметь хотя бы один тег. Пустой массив отключает
условие.

### tags_all

Array of strings. Модель должна иметь все теги. Пустой массив отключает условие.

### exclude_tags

Array of strings. Совпадение любого тега исключает модель. Все tag-сравнения
точные и регистронезависимые.

### model_id_regex

String с Go/RE2 regexp для полного `namespace/repo`. Пустая строка отключает
условие. Lookaround и backreferences не поддерживаются RE2.

### min_parameters_b

Number или null. Нижняя включительная граница effective parameters в миллиардах.

### max_parameters_b

Number или null. Верхняя включительная граница. Min не может превышать max.

### include_unknown_parameters

Boolean. Имеет эффект при заданном min/max:

- `false` — неизвестный размер исключается;
- `true` — неизвестный размер проходит parameter filter.

Default: false.

### min_downloads и max_downloads

Integer, включительные границы. `0` отключает соответствующую границу.

### min_likes и max_likes

Integer, включительные границы. `0` отключает соответствующую границу.

### sort_by

String: `downloads`, `likes`, `created_at`, `parameters` или `repo_id`.

Default: `downloads`.

### sort_direction

String: `asc` или `desc`. Default: `desc`. При равенстве tie-breaker — `repo_id`
по возрастанию.

### limit

Неотрицательный integer. `0` сохраняет все совпадения; положительное значение
оставляет первые N после фильтров и сортировки.

Default: 0.

### compute_profiles

Array имён профилей. Неизвестное имя вызывает ошибку. Пустой массив отключает
compute estimates.

## 8. Классификация model_kind

Приоритет:

1. `baseModels.relation`: adapter, finetune, quantized, merge.
2. Quantization tags: gguf, gptq, awq, exl2, quantized.
3. Merge tags: model-merge, merge.
4. Adapter tags/relation tags/имя: PEFT, LoRA, QLoRA, adapter.
5. Fine-tune relation tags/имя: finetune, fine-tune, SFT, DPO.
6. Если признаков нет — base.

`base` означает отсутствие обнаруженных признаков производной модели, а не
доказательство обучения с нуля.

## 9. Семантика parameter count

- `own_parameters` — собственный `safetensors.total` repo.
- `effective_parameters` — размер для фильтра и compute estimate.
- `parameters_b` — effective parameters / 1e9.

Для base и full fine-tune effective равен own. Для adapter при включённом
разрешении базы используется `safetensors.total` base model. Если база не указана
или недоступна, используется own adapter count.

Весовой репозиторий без Safetensors может иметь неизвестный размер.

## 10. Порядок применения

1. Вычисляется самая ранняя дата selections.
2. `/api/models` сканируется по `createdAt` от новых к старым.
3. Применяются общая дата и `require_weights`.
4. Разрешаются параметры нужных adapter bases.
5. Для selection применяются дата, pipeline, library, kind, tags, regexp,
   popularity и parameter filters.
6. Выполняется сортировка.
7. Без `owner_types` применяется `limit`.
8. Загружаются уникальные owner profiles.
9. Применяется `owner_types`, затем limit для таких selections.
10. Считаются owner aggregates и пишутся результаты.

## 11. Выходные файлы

### models.csv

Колонки:

- `selection`, `repo_id`, `repo_url`;
- `owner`, `owner_type`;
- `created_at`, `last_modified`;
- `pipeline_tag`, `library_name`, `model_kind`, `base_model`;
- `own_parameters`, `effective_parameters`, `parameters_b`;
- `downloads`, `likes`;
- `tags` с разделителем `|`;
- `compute_estimates_json` — JSON-массив внутри CSV-ячейки.
- `errors` — нефатальные проблемы конкретной записи, разделённые ` | `.

Одна модель может иметь несколько строк, если входит в несколько selections.

### models.json

Те же записи. Compute estimates являются нормальным вложенным массивом.

### owners.csv

Колонки:

- `owner`, `owner_type`, `profile_url`, `fullname`;
- `verified`, `pro`, `plan`, `followers`;
- `models`, `datasets`, `spaces`, `papers`, `members`;
- `created_at`, `details`;
- `selected_repos`, `selected_downloads`, `selected_likes`;
- `selections` с разделителем `|`;
- `error`.

Owner aggregates дедуплицируются по repo, даже если repo в нескольких selections.

### owners.json

Те же profile и aggregate fields в JSON.

### summary.json

Корневые поля:

- `generated_at`, `effective_now`, `complete`;
- `pages_scanned`, `models_scanned`;
- `selections`.

Каждый summary selection содержит `description`, `models`, `unique_owners`,
`owner_types`, `model_kinds`, `known_parameters`, `total_downloads`, `total_likes`.
Selections с нулём совпадений тоже присутствуют.

### catalog.log.jsonl

Структурированный журнал запуска. Имя и местоположение меняются через
`logging.file`. Токен `HF_TOKEN` в события не записывается.

## 11A. Что происходит при ошибках

Фатальные ошибки завершают процесс с ненулевым exit code. К ним относятся:
невалидный JSON или неизвестная опция, некорректные диапазоны, невозможность
получить или декодировать страницу каталога, отмена context, а также ошибка
создания, кодирования, записи, flush, sync или закрытия выходного файла.

Восстановимые ошибки не обрывают всю многотысячную выборку:

- невалидный `createdAt` отбрасывает конкретную модель и создаёт `warn`;
- неудачное получение параметров base-модели создаёт `warn`, а причина попадает
  в `models.json` и колонку `errors`, если запись допускает неизвестный размер;
- неудачный owner enrichment создаёт `warn`, оставляет тип `unknown` и сохраняет
  причину в поле/колонке owner `error`;
- временные HTTP-ошибки повторяются с backoff; каждая повторная попытка имеет
  событие `http_retry`, исчерпание попыток — `http_retries_exhausted`.

Неожиданная panic перехватывается на границе catalog-run, сохраняется вместе со
stack trace как событие `panic`, после чего процесс возвращает ошибку.

## 12. Complete и воспроизводимость

Для финальной выборки требуется `complete: true`. `false` означает остановку до
нижней временной границы, обычно из-за `max_pages` или прерывания.

Для повторяемого окна фиксируйте `scan.now`. Downloads, likes и owner counters
могут меняться независимо от фиксированного периода.

## 13. Практические selections

Готовый большой набор находится в `catalog.scenarios.json`. Он содержит popular
LLM, отдельные base/fine-tune/adapter выборки, диапазоны 1–3B, 7–14B, 30–40B и
64–128B, разрезы user/organization, code-модели, календарный 2025 год и четыре
варианта кластера.

Полный запуск:

```bash
go run . catalog --config catalog.scenarios.json
```

Для эксперимента скопируйте файл, удалите ненужные элементы из `selections` и
задайте отдельный `output_dir`. Все selections выполняются за один общий проход
Hub, поэтому десяток сценариев не означает десяток полных скачиваний каталога.

### Top-100 LLM за год

```json
{
  "name": "popular_llm_last_year",
  "lookback_days": 365,
  "pipeline_tags": ["text-generation", "text2text-generation"],
  "model_kinds": ["base", "finetune", "adapter"],
  "exclude_tags": ["gguf", "gptq", "awq", "exl2", "model-merge"],
  "sort_by": "downloads",
  "sort_direction": "desc",
  "limit": 100
}
```

### Base LLM

```json
{
  "name": "base_llm",
  "lookback_days": 365,
  "pipeline_tags": ["text-generation", "text2text-generation"],
  "model_kinds": ["base"],
  "sort_by": "downloads",
  "limit": 100
}
```

### Adapters

```json
{
  "name": "adapters",
  "lookback_days": 365,
  "pipeline_tags": ["text-generation", "text2text-generation"],
  "model_kinds": ["adapter"],
  "sort_by": "downloads",
  "limit": 100
}
```

### Все 64–128B LLM

```json
{
  "name": "llm_64_128b",
  "lookback_days": 365,
  "pipeline_tags": ["text-generation", "text2text-generation"],
  "model_kinds": ["base", "finetune", "adapter"],
  "min_parameters_b": 64,
  "max_parameters_b": 128,
  "include_unknown_parameters": false,
  "sort_by": "downloads",
  "limit": 0,
  "compute_profiles": ["h100_1000x1", "h100_256x4"]
}
```

### Только организации с 10 000+ downloads

```json
{
  "name": "popular_org_models",
  "lookback_days": 365,
  "owner_types": ["organization"],
  "min_downloads": 10000,
  "sort_by": "downloads",
  "limit": 100
}
```

## 14. Проверка и smoke run

```bash
python3 -m json.tool catalog.example.json >/dev/null
```

Для smoke run задать `scan.max_pages: 1` и выполнить:

```bash
go run . catalog --config catalog.example.json --output smoke-results
```

Для полного запуска вернуть `max_pages: 0` и проверить `complete: true`.

## 15. Старый scratch-counter CLI

Запуск без `catalog` включает старый режим:

```bash
go run . [flags]
```

Все его флаги:

- `--years <comma-separated>` — UTC-годы. Default: `2025,2026`.
- `--output <path>` — директория. Default: `results`.
- `--endpoint <url>` — Hub endpoint. Default: `https://huggingface.co`.
- `--token <token>` — токен; fallback на `HF_TOKEN`.
- `--page-size <1..1000>` — API page size. Default: 1000.
- `--workers <1..64>` — README workers. Default: 8.
- `--timeout <duration>` — Go duration, например `30s` или `2m`. Default: `30s`.
- `--retries <n>` — число повторов. Default: 5.
- `--max-pages <n>` — 0 означает полный проход. Default: 0.
- `--quiet` — скрыть прогресс.
- `--help`, `-h` — справка.

```bash
go run . --years 2025,2026 --workers 16 --timeout 45s --output scratch-results
```
