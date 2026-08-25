# Catalog mode: полный reference

Документ описывает режим выборок моделей и владельцев. Старый режим подсчёта
scratch-моделей рассмотрен отдельно в конце.

## 1. Запуск

```bash
go run ./cmd/hfscraper catalog [flags]
```

Доступные CLI-флаги:

- `--config <path>` — JSON-конфиг. Default: `configs/catalog.example.json`.
- `--output <path>` — переопределяет `output_dir` из JSON.
- `--help`, `-h` — справка.

Примеры:

```bash
go run ./cmd/hfscraper catalog
go run ./cmd/hfscraper catalog --config configs/llm.json --output results/llm
HF_TOKEN=hf_... go run ./cmd/hfscraper catalog --config configs/catalog.example.json
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

## 3. Compute-профили

Поле `compute_profiles` содержит массив профилей с описанием GPU кластеров и настроек вычислений.

Каждый профиль может содержать следующие поля:

### Обязательные поля:
- `name` — уникальное имя профиля
- `gpu_name` — название GPU (например, "NVIDIA H100")
- `gpu_tflops` — производительность GPU в TFLOPS 
- `efficiency` — эффективность вычислений от 0 до 1
- `gpu_hour_cost_usd` — стоимость одного GPU-часа в USD
- `tokens_per_parameter` — количество токенов на параметр (для formula.txt)
- `machines` — число машин в кластере
- `gpus_per_machine` — число GPU на машину
- `finetune_compute_fraction` — доля расходов на fine-tuning

### Необязательные поля для diffusion-моделей:
- `diffusion_image_budget` — количество изображений на параметр (для оценки стоимости обучения diffusion моделей)
- `latent_sequence_length` — латентная длина последовательности (для оценки стоимости обучения diffusion моделей)

Профиль считается включённым для diffusion, если оба поля `diffusion_image_budget` и `latent_sequence_length` > 0.

## 4. Выборки

Поле `selections` содержит массив выборок с параметрами фильтрации и профилей вычислений:

- `compute_profiles` — список имён compute-профилей, которые будут использованы для оценки стоимости
- `pipeline_tags` — фильтр по pipeline тегам (например, ["text-to-image", "image-to-image"])
- `model_kinds` — фильтр по типу модели
- `target_llm_only` — если true, исключаются non-text модели

## 5. Пример конфига с diffusion профилями

```json
{
  "compute_profiles": [
    {
      "name": "h100_1000x1",
      "description": "1000 машин по одной H100",
      "gpu_name": "NVIDIA H100",
      "gpu_tflops": 989,
      "efficiency": 0.4,
      "gpu_hour_cost_usd": 1.85,
      "tokens_per_parameter": 20,
      "machines": 1000,
      "gpus_per_machine": 1,
      "finetune_compute_fraction": 0.05
    },
    {
      "name": "diffusion_conservative",
      "description": "Консервативный бюджет для diffusion-моделей",
      "gpu_name": "NVIDIA H100",
      "gpu_tflops": 989,
      "efficiency": 0.4,
      "gpu_hour_cost_usd": 1.85,
      "tokens_per_parameter": 20,
      "machines": 1000,
      "gpus_per_machine": 1,
      "finetune_compute_fraction": 0.05,
      "diffusion_image_budget": 3.0,
      "latent_sequence_length": 1024
    }
  ]
}
```

## 6. Результаты

Результаты содержат информацию о модели, включая:
- `training_cost_usd` — стоимость обучения (если определена)
- `training_cost_method` — метод оценки стоимости 
- `compute_estimates` — список всех рассчитанных оценок с профилями

Для diffusion моделей методы будут иметь префикс `formula_txt_diffusion_`, что обеспечивает автоматическую дедупликацию.

## 7. Дополнительные примечания

- Поля `diffusion_image_budget` и `latent_sequence_length` активны только для профилей с обоими значениями > 0
- Все оценки, созданные по этим профилям, помечаются флагом `assumed_budget: true`
- Оценка стоимости для diffusion моделей выполняется по формуле: `6 × N_eff × (N_total × бюджет)`
