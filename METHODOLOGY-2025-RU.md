# Стоимость обучения публичных text LLM на Hugging Face за 2025 год

## Итог

Основная evidence-adjusted оценка: **$50,545,833.34**.

Это H100 GPU-rental-equivalent по ставке $1.85/GPU-час, а не сумма счетов
компаний. В неё вошли 1,094 дедуплицированных training run:

| Категория | Репозиториев | Run с cost | Стоимость, USD |
|---|---:|---:|---:|
| Base | 94,129 | 545 | $48,699,290.19 |
| Fine-tune / CPT / post-training | 142,304 | 334 | $1,841,879.74 |
| Adapter / LoRA / QLoRA | 51,504 | 215 | $4,663.41 |
| Fork / abliterated / checkpoint | 28,895 | 0 | $0 |
| Quantized / conversion | 119,382 | 0 | $0 |
| Merge | 11,319 | 0 | $0 |
| **Все целевые text LLM** | **447,533** | **1,094** | **$50,545,833.34** |

Неокруглённое авторитетное значение в `summary.json`:
`50545833.34489301` USD.

## Почему не 108 и не 126 тысяч

126 тысяч в старом запуске были остаточным классом репозиториев, а не числом
самостоятельно обученных моделей. Название `base`, отсутствие `base_model` и
наличие весов не доказывают новый pretraining: в остатке находятся переименованные
чекпойнты, форки, копии и карточки с недостаточной metadata.

Новая полная воронка:

1. Просмотрено 1,846,000 репозиториев, созданных в 2025 UTC.
2. У 902,093 найдены распознанные веса.
3. В text-LLM рынок вошли 447,533 репозитория; image, video, diffusion, VLM,
   audio/speech, robotics, protein/genomic и другие non-text модели исключены.
4. В остаточном base-классе осталось 94,129 репозиториев.
5. Полные model cards всех кандидатов прочитаны из bulk Parquet. У 3,958 найдены
   признаки независимого pretraining или явное заявление base-модели.
6. После проверки параметров, календаря, lineage и дедупликации стоимость
   получили 545 независимых base-run.
7. Из 287,952 trainable derivative-кандидатов опубликованный compute/cost найден
   у 699; после календаря и дедупликации в сумму вошли 334 fine-tune и 215
   adapter-run.

То есть 545 — не выборка из 94 тысяч названий. Это результат проверки карточек
всех кандидатов. Репозитории без доказательств сохранены в `models.json` с
пустым `training_cost_usd`, а не удалены из рынка.

## Таксономия

Целевые модели — публичные репозитории с весами для natural-language text tasks:
text generation, conversational, text2text и fill-mask, плюс известные LLM
families. Наличие скачиваемых весов означает open-weight; это не гарантирует
OSI open-source лицензию.

Сначала выделяются adapter/LoRA/QLoRA, fine-tune/SFT/DPO/CPT/distillation,
quantization/conversion, merge и fork. К форкам относятся abliterated,
uncensored, repack/reupload/mirror, qwenified/adjusted/pruned, swarm-artifacts,
checkpoints и step snapshots. Они остаются отдельными рыночными подкатегориями,
но не получают повторную стоимость полного pretraining.

Разные размеры одной настоящей base-family считаются отдельно: например, 8B,
14B и 30B — отдельные training run. Датированные snapshots одной накопительной
траектории, точные копии card/run, checkpoint-series и cross-owner mirrors
схлопываются.

## Формула и основной estimate

Из `formula.txt`:

```text
FLOPs = 6 × N × T
H100 throughput = 989 × 10^12 FLOP/s
training efficiency = 0.4
price = $1.85 / GPU-hour
```

Для dense-модели при fallback `T = 20 × N`:

```text
GPU-hours = 84.2601954836535 × P_B²
cost USD  = 155.881361644759 × P_B²
```

Основной estimate использует более сильные публичные данные:

- если base-card сообщает фактический объём pretraining tokens, он заменяет
  fallback `20 × N`;
- для MoE FLOPs/token считаются по опубликованным active-параметрам, а объём
  токенов — по reported tokens либо fallback от total-параметров;
- если этих данных нет, применяется формула `155.881361644759 × P_B²`;
- для fine-tune/adapter формула полного pretraining не применяется: учитываются
  только direct reported cost, aggregate GPU-hours либо wall time × GPU count;
- при отсутствии цены reported compute умножается на $1.85; при отсутствии
  типа GPU подразумевается одна H100 и это отмечается флагами `assumed_*`.

Сумма считается как сумма независимых run. Нельзя сначала сложить параметры
разных моделей, а затем возвести сумму в квадрат.

## Sensitivity: три воспроизводимых сценария

| Сценарий | Total, USD | Интерпретация |
|---|---:|---|
| **Evidence-adjusted, основной** | **$50,545,833.34** | Reported pretraining tokens и MoE active count имеют приоритет |
| Строго 20 tokens/parameter, MoE `total × active` | $20,571,760.43 | Не использует reported token budgets |
| Буквальный `formula.txt`, MoE также `total²` | $271,571,788.26 | Математический upper-сценарий для sparse models |

Последний сценарий резко выше из-за trillion-parameter MoE. Например,
`moonshotai/Kimi-K2-Base` имеет около 1.026T total, но 32B active parameters.
Считать каждый total-параметр активным на каждом токене физически неверно;
поэтому `total²` сохранён как sensitivity, а не выбран headline.

Основной результат выше сценария 20 tokens/parameter, потому что ряд публичных
моделей сообщает существенно больший token budget: Qwen3 — 36T, Apertus — 15T,
MiMo — 25T и т.д.

## Base по размеру

| Параметры | Base-репозиториев | Run с cost | Основная стоимость, USD |
|---|---:|---:|---:|
| <2B | 47,335 | 418 | $1,273,803.37 |
| 2–<7B | 10,840 | 47 | $3,025,085.85 |
| 7–<13B | 11,495 | 37 | $8,348,937.65 |
| 13–<34B | 5,279 | 23 | $11,716,336.66 |
| 34–<70B | 75 | 8 | $2,270,757.61 |
| 70–<120B | 371 | 6 | $11,110,221.92 |
| 120–<500B | 94 | 4 | $4,418,596.43 |
| >=500B | 65 | 2 | $6,535,550.71 |
| Unknown | 18,575 | 0 | $0 |

Это интервалы репозиториев, а не утверждение, что каждая строка в `base`
является новым обучением. Costed column показывает прошедшие evidence-фильтр run.

## Календарь и дедупликация

`createdAt` репозитория используется как календарный proxy, когда дата обучения
не опубликована. Явный `Training Dates: ...` имеет приоритет: run с датами только
вне 2025 не начисляется 2025 году. Это неизбежная нижняя/верхняя неопределённость,
поскольку Hugging Face не имеет обязательного `training_completed_at`.

Дедупликация использует normalized card hash, training-evidence/run hash,
параметры, owner/family, checkpoint suffix, base lineage и earliest publication.
Для датированных cumulative snapshots остаётся самый полный disclosed run.
Глобального training-run ID на Hub нет, поэтому полностью переписанное зеркало
теоретически может остаться несклеенным.

## Что не входит

- приватные кластеры и закрытые модели без публичных весов (Fable, Codex и т.п.);
- удалённые или приватные HF-репозитории;
- image/video/diffusion/VLM/audio и прочие non-text модели;
- forks, merges, quantizations и conversions как новое обучение;
- fine-tune/adapters без опубликованного compute/cost;
- base без известных параметров или без достаточного pretraining evidence;
- CPU/host/network/storage, датасеты, зарплаты, электроэнергия вне rental rate,
  неудачные эксперименты и inference.

Все reported values являются заявлениями model-card и автоматически не
подтверждаются счетами. Даже когда карточка указывает A100/TPU/RTX, при отсутствии
собственной ставки применяется заданный пользователем H100-equivalent rate
$1.85. Поэтому корректное название результата — оценка публично наблюдаемого
training compute, не бухгалтерские мировые расходы.

## Владельцы

Namespace сохранён для всех 447,533 строк и агрегирован в `owners.json/csv`.
Без HF-токена production-конфиг не делает 64,688 дополнительных profile-запросов,
поэтому `owner_type` остаётся `unknown`. Имена и суммы владельцев корректны;
автоматически разделить user и organization без rate-limited enrichment нельзя.

## Воспроизведение без HF-токена

```bash
go test -count=1 ./...
go run . catalog --config catalog.2025-llm-market.json
```

Повторный запуск использует gzip-checkpoint каталога и пять локальных Parquet-
шардов model cards в `results-2025-production/cache/`. Индивидуальные README не
скачиваются. Основные результаты:

```text
results-2025-llm-market/report.md
results-2025-llm-market/summary.json
results-2025-llm-market/models.json
results-2025-llm-market/models.csv
results-2025-llm-market/owners.json
results-2025-llm-market/owners.csv
```

`summary.json` и `models.json` содержат неокруглённые значения и являются
источником точной арифметики. CSV округляет каждую строку до центов.

## Почему прежние числа были неверны

- ~$21.9B и более: P² начислялось остаточному `base`, включая форки, mirrors,
  conversions и модели без доказанного независимого pretraining.
- ~$3.64M: параметрическая стоимость base была целиком отключена.
- 108 моделей: принималась только узкая буквальная фраза `trained from scratch`;
  pretraining sections, declared official base families и reported tokens были
  потеряны.
- ~$51.07M: до финального аудита `RTX 4090` читалось как 4,090 GPU-hours,
  `4.82 TPUv3-8 Hours` как 656 GPU-hours, а две версии AstroSage считались дважды.

Текущий результат построен полным проходом bulk-card snapshot, а перечисленные
ошибки закреплены regression-тестами.
