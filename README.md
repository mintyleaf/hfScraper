# Hugging Face models trained from scratch counter

Скрипт считает по годам репозитории моделей Hugging Face, авторы которых **явно
утверждают в README**, что модель обучена с нуля. По умолчанию анализируются 2025
и 2026 годы.

## Важное ограничение

У Hugging Face нет обязательного и достоверного поля `trained_from_scratch`.
Поэтому получить абсолютно точное число невозможно. Скрипт выдаёт две метрики:

- `confirmed` — есть файл весов, нет признаков производной модели и в README есть
  явная ненегированная фраза об обучении с нуля;
- `broad_upper_bound` — `confirmed` плюс `candidate`: модели с весами, для которых
  не найдено признаков fine-tune/adapter/merge/quantization, но и нет явного
  подтверждения обучения с нуля.

Год означает UTC-год поля `createdAt` репозитория. Это год публикации репозитория,
а не доказанная дата завершения обучения. Удалённые и приватные репозитории, а
также модели вне Hugging Face в подсчёт не попадут.

## Запуск

Требуется Go 1.24.9+.

```bash
go run ./cmd/hfscraper --years 2025,2026 --output results
```

Короткая проверка только первой страницы (это **не итоговый подсчёт**):

```bash
go run ./cmd/hfscraper --page-size 20 --max-pages 1 --output smoke-results
```

В таком отчёте `complete` будет `false`.

Для gated-репозиториев можно передать read-only токен через переменную окружения:

```bash
HF_TOKEN=hf_... go run ./cmd/hfscraper
```

Полный обход может занять много времени: API просматривается от новых моделей к
старым, а README кандидатов загружаются отдельно. При `429` и временных ошибках
работает exponential backoff. Прогресс печатается после каждой страницы.

Результаты:

- `results/summary.json` — итоговые числа и разбивка причин;
- `results/models.csv` — аудит каждой рассмотренной модели, статус и фрагмент
  текста, послуживший подтверждением.

Если процесс прервать, частичные файлы сохранятся, но следующий запуск начинает
обход заново и перезаписывает их. Для более строгой проверки отфильтруйте в CSV
строки `status=confirmed` и вручную просмотрите поле `evidence`.

## Проверка

```bash
go test ./...
go run ./cmd/hfscraper --help
```

Алгоритм сначала исключает репозитории без распознанного файла весов, с
`base_model`, adapter/LoRA, merge и распространёнными форматами квантованных
конверсий. Распознаются веса Transformers и Diffusers с любыми именами файлов,
а также PyTorch, TensorFlow/Keras, Flax/JAX, ONNX, TFLite, Core ML, Paddle,
MXNet, scikit-learn и некоторые экспортные форматы. Файлы токенизатора и состояния
тренера (`tokenizer.bin`, `optimizer.pt`, `training_args.bin` и подобные) весами
не считаются. Затем README оставшихся репозиториев проверяется на явные формулировки
`trained/pretrained from scratch`, `trained from random initialization` и их
несколько языковых вариантов. Все эвристики находятся в начале
`cmd/hfscraper/scratch.go` и легко расширяются.

## Структура репозитория

```text
cmd/hfscraper/  CLI и логика скрейпера
configs/        готовые конфигурации catalog-режима
docs/           инструкции, методология и формула расчёта
reports/        исследовательские отчёты
results/        сохранённые результаты запусков
```

## Каталог моделей и владельцев

### Стратифицированная оценка text + diffusion

Воспроизводимая выборка с полным census моделей от 34B, bounded README-fetch и локальной LLM описана в [docs/STRATIFIED-MARKET-2025-RU.md](docs/STRATIFIED-MARKET-2025-RU.md).

Безопасный plan-only запуск не использует сеть, LLM и не считает стоимость:

```bash
go run ./cmd/hfscraper market-sample --config configs/catalog.2025-stratified-market.local-llm.json
```

Боевой запуск требует явного флага `--execute`:

```bash
HF_LLM_BASE_URL=http://127.0.0.1:11434/v1 HF_LLM_MODEL=qwen2.5:14b go run ./cmd/hfscraper market-sample --config configs/catalog.2025-stratified-market.local-llm.json --execute
```

Для настраиваемых выборок по популярности, размеру, типу модели и владельцу
используйте режим `catalog`:

```bash
go run ./cmd/hfscraper catalog --config configs/catalog.example.json
```

Он сохраняет модели и профили владельцев в CSV и JSON. Поддерживаются несколько
выборок за один проход API, base/fine-tune/adapter-фильтры, диапазоны параметров,
периоды, теги, pipeline, сортировка, лимиты и расчёт времени обучения на разных
конфигурациях кластера. Полное описание: [docs/CATALOG.md](docs/CATALOG.md).

Совсем пошаговая инструкция для первого запуска: [docs/HOW-TO-RU.md](docs/HOW-TO-RU.md).

### Боевой расчёт стоимости за 2025 год

Подробное описание источников, формул, дедупликации, аудита и ограничений:
[docs/LOCAL-LLM-2025-RU.md](docs/LOCAL-LLM-2025-RU.md). Старый строгий проход
сохранён отдельно как [архив](docs/METHODOLOGY-2025-RU.md).

Пошаговый документ для согласования процесса с заказчиком:
[docs/PROCESS-2025-CUSTOMER-APPROVAL-RU.md](docs/PROCESS-2025-CUSTOMER-APPROVAL-RU.md).

Старый строгий результат на 1,094 costed-репозитория не является оценкой всего
рынка: он отражал только строки, которые прошли узкие регулярные выражения.
Используйте новый проход с локальной LLM, без popularity-порогов. Он выводит
аудируемые lower, central и upper суммы и отдельно показывает число всех
репозиториев и дедуплицированных training runs.

Рекомендуемый боевой конфиг без ограничения размера модели:

```bash
HF_LLM_BASE_URL=http://127.0.0.1:11434/v1 HF_LLM_MODEL=qwen2.5:7b go run ./cmd/hfscraper catalog --config configs/catalog.2025-llm-market.local-llm.json
```

Срочный совместный проход text + diffusion без локальной LLM:

```bash
go run ./cmd/hfscraper catalog --config configs/catalog.2025-text-and-diffusion.no-llm.json
```

Только быстрый API-массив preliminary base для обеих выборок, без Parquet и без
LLM-нагрузки:

```bash
go run ./cmd/hfscraper catalog --config configs/catalog.2025-text-and-diffusion.no-llm.json --pre-llm-only
```

Отдельный diffusion-проход с той же local-LLM lineage-проверкой:

```bash
HF_LLM_BASE_URL=http://127.0.0.1:11434/v1 HF_LLM_MODEL=qwen2.5:7b go run ./cmd/hfscraper catalog --config configs/catalog.2025-diffusion-market.local-llm.json
```

До model-card/LLM этапа каждый проход создаёт `pre-llm-summary.json` и отдельные
JSON/CSV-массивы всех preliminary base-кандидатов. Год задаётся только
`created_from`/`created_to` в selection; 2025 — значение текущих отчётных
конфигов, а не ограничение алгоритма.

Старый `configs/catalog.2025-llm-market.json` оставлен только для воспроизводимости
строгого regex-only подхода.

Для base-моделей уравнение `6 × N × T` из `docs/formula.txt` применяется только при
подтверждении независимого pretraining и известных параметрах. Reported token
budget имеет приоритет, иначе используется fallback 20 токенов/параметр. Для
MoE учитываются опубликованные active parameters. Остаточный класс `base` сам по
себе подтверждением не считается. Fork, quantized/conversion и merge не получают
стоимость; fine-tune/adapter учитываются только при reported compute/cost.
Dense fallback после подстановки констант:

```text
training_cost_USD = 155.881361644759 × parameters_billions²
```

Merge, quantization и форматы-конверсии исключаются из стоимости. Копии
карточек, старые зеркала и повторные публикации одного training run
дедуплицируются. Исходный
evidence, hashes и даты сохраняются для аудита.

Боевой конфиг не загружает README по одному. Он один раз скачивает пять
Parquet-шардов `librarian-bots/model_cards_with_metadata` с полными текстами
model cards (около 1.28 ГБ суммарно на момент написания), сохраняет их в
`results-2025-production/cache/` и затем ищет compute локально. Незавершённая
загрузка продолжается из файла `.part`. Полный проход
каталога также сохраняется как gzip-checkpoint, поэтому повторный запуск не
сканирует весь Hub заново. В новом local-LLM конфиге user/organization-профили
запрашиваются только для владельцев, вошедших в central estimate; остальные
сохраняются с типом `unknown`, чтобы не добавлять десятки тысяч API-запросов.

Главные результаты нового прохода находятся в
`results-2025-llm-market-local-llm/models.csv`,
`owners.csv`, `summary.json` и готовом `report.md`. Поля `training_cost_usd`
содержат сумму в долларах, а `training_cost_method` отмечает расчёт scratch по
`docs/formula.txt`.

Результат является formula-equivalent оценкой H100 compute, а не бухгалтерской
суммой: Hub не требует публиковать фактическое железо, длительность или счета.
