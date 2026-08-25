# Расчёт стоимости публичных text LLM за 2025 год

## Что именно считает этот запуск

Главный результат — оценка H100-equivalent стоимости вычислений для обучения
публично доступных text LLM, репозитории которых появились на Hugging Face в
2025 календарном году. Это не сумма счетов компаний: Hugging Face не хранит
обязательные сведения о фактической цене кластера, а дата окончания обучения
часто не опубликована.

Запуск считает только текстовые LLM. Image/video/diffusion/VLM/audio-модели
жёстко исключены и не попадают ни в число моделей, ни в стоимость. Приватные и
удалённые репозитории также недоступны этому снимку.
Fork/mirror, merge, quantization и конверсия форматов остаются в статистике
рынка, но не получают стоимость нового обучения.

## Воронка

1. Каталог Hub отбирается по `createdAt` с `2025-01-01` по `2025-12-31` UTC.
   Нужны опубликованные веса. Токен Hugging Face не обязателен.
2. По pipeline, tags и metadata отделяются text LLM от image/video/audio/VLM и
   других нецелевых моделей.
3. Явные adapter/LoRA/QLoRA, fine-tune/CPT/SFT/DPO/RLHF, fork, merge и
   quantized/conversion сразу попадают в свои категории. Остаток — только
   base-кандидаты, а не автоматически подтверждённые новые training runs.
4. Пять Parquet-шардов `librarian-bots/model_cards_with_metadata` скачиваются
   один раз. Все карточки читаются локально; отдельных запросов README для
   десятков тысяч репозиториев нет.
5. Base-кандидаты с известным числом параметров проверяет локальная LLM через
   OpenAI-compatible API. Она выбирает один класс:
   `independent_base`, `continued_pretraining`, `finetune`, `adapter`,
   `fork_or_mirror`, `quantized`, `merge`, `non_text` или `unknown`.
6. Решение high/medium принимается только вместе с короткой дословной цитатой,
   которая реально присутствует в model card. Год берётся не из догадки LLM, а
   только из этой цитаты или из детерминированно распознанной строки training
   dates. Если год обучения не опубликован, 2025 считается по `createdAt` и это
   явно остаётся допущением.
7. Совпавшие card/evidence hashes, checkpoint-траектории, одинаковые base-family
   mirrors и одинаковый `canonical_training_run` с точным совпадением числа
   параметров считаются один раз. Разные размеры одной семьи не склеиваются.

Слово `base` в имени репозитория больше не является достаточным доказательством
обучения с нуля. Порогов по likes/downloads в этом конфиге нет: непопулярная
модель имеет те же правила, что и популярная.

## Формула стоимости

Константы взяты из [`formula.txt`](formula.txt): H100 989 TFLOPS, 40% полезной
эффективности и $1.85 за H100 GPU-hour.

```text
T_fallback = 20 × N_total
FLOPs      = 6 × N_effective × T
GPU_hours  = FLOPs / (989 × 10^12 × 3600 × 0.4)
cost_USD   = GPU_hours × 1.85
```

Для dense base-модели без опубликованного token budget это:

```text
cost_USD = 155.881361644759 × parameters_billions²
```

Если карточка явно сообщает число pretraining tokens, оно заменяет fallback.
Для MoE `N_effective` — опубликованное active/activated число параметров, а
token fallback всё равно считается от total parameters. `summary.json` также
сохраняет буквальный total² sensitivity-сценарий, но он не является основной
оценкой.

Для fine-tune и adapters стоимость включается только когда карточка сообщает
собственный training time/число GPU или прямую стоимость. Если ставка не
указана, применяется $1.85/GPU-hour; опубликованная ставка имеет приоритет.
Если compute time отсутствует, стоимость не выдумывается.

## Почему в отчёте три суммы

- `lower_training_cost_usd`: только детерминированное independent-pretraining
  evidence и опубликованный compute производных моделей.
- `total_training_cost_usd`: основной central estimate — lower плюс
  high-confidence решения локальной LLM.
- `upper_training_cost_usd`: central плюс medium-confidence независимые
  training runs.

Для рынка следует цитировать central estimate вместе с диапазоном lower–upper и
количеством `run` в каждой строке. Число всех репозиториев нельзя называть
числом обученных моделей: один training run может иметь много публикаций,
квантовок и зеркал.

## Запуск без HF-токена

Нужны Go 1.24.9+, около 1.3 ГБ места под Parquet и локальный сервер с
OpenAI-compatible `/v1/models` и `/v1/chat/completions`.

Простой вариант с Ollama (в первом терминале):

```bash
ollama pull qwen2.5:7b
ollama serve
```

Если Ollama уже работает как сервис, второй шаг не нужен. Боевой запуск во
втором терминале:

```bash
HF_LLM_BASE_URL=http://127.0.0.1:11434/v1 HF_LLM_MODEL=qwen2.5:7b go run ./cmd/hfscraper catalog --config configs/catalog.2025-llm-market.local-llm.json
```

`HF_TOKEN` не нужен. Для другого сервера достаточно заменить URL и имя модели.
Если сервер требует ключ, задайте имя переменной в `local_llm.api_key_env`
конфига и положите ключ в эту переменную окружения.

Первый запуск сначала использует или создаёт каталог-checkpoint, затем один раз
скачивает Parquet. Локальные решения дописываются в
`results-2025-llm-market-local-llm/cache/local-llm-review-v1.jsonl`. После
`Ctrl+C` запустите ту же команду: уже сохранённые решения будут взяты из кэша.
Прогресс печатается каждые 100 уникальных LLM-вызовов.

Не меняйте модель или prompt в середине одного анализа. Версия prompt и hash
карточки входят в cache key; изменение встроенного prompt автоматически
создаёт другие ключи.

## Результаты и аудит

После завершения смотрите:

- `results-2025-llm-market-local-llm/report.md` — читаемый итог и владельцы;
- `summary.json` — суммы, воронка, категории и confidence;
- `models.csv` / `models.json` — каждая строка, evidence, tier, метод и ошибки;
- `owners.csv` / `owners.json` — пользователи/организации и central subtotal;
- `catalog.log.jsonl` — полный журнал и прогресс.

В `models.csv` поля `lower_training_cost_usd`, `training_cost_usd` и
`upper_training_cost_usd` позволяют воспроизвести все три суммы. Поле
`local_llm_review_json` хранит класс, confidence, цитату и canonical run.
Чтобы не делать десятки тысяч дополнительных Hub-запросов, профиль
user/organization запрашивается только для владельцев, вошедших в central
estimate. Остальные владельцы сохраняются с типом `unknown`; их репозитории из
рыночной воронки не исчезают.

## Проверка перед боевым запуском

```bash
go test -count=1 ./...
go vet ./...
```

Сам скрипт делает completion-preflight до массовой классификации. Неверный URL,
несуществующая модель или несовместимый JSON-ответ останавливают запуск сразу,
а не после десятков тысяч карточек.
