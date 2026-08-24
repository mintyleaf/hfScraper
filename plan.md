## Plan: Diffusion-затраты через сценарии-профили (hfScraper SAM)

**TL;DR.** Diffusion-затраты как два именованных compute-профиля-сценария (консервативный/верхний уровень бюджета) в `catalog.scenarios.json` по образцу MoE-sensitivity; reported compute сохраняет приоритет, scratch-классификатор и дедупликация не трогаются. Scope согласован: text2img + img2img + text-to-video; assumption-оценки только с явными флагами и отдельным бакетом методов. Move-only разбивка `catalog.go` на тематические файлы пакета `main` уже выполнена и верифицирована (build/vet green, тесты 112/112); поля бюджета и их валидация (A1) применены во внешнем источнике — build exit 0.

**Контекст-факты (из кода/конфигов)**
- Один пакет `main`; тематические файлы уже на месте: `catalog_config.go`, `catalog_selection.go`, `catalog_kind.go`, `catalog_record.go`, `catalog_dedup.go`, `catalog_compute.go`, `catalog_run.go`.
- Механизм «разные допущения = разные профили» существует (`computeProfile`, selections ссылаются по имени) — расширяем, не изобретаем новое.
- Арифметика: `estimateCompute`/`estimateBaseTrainingCompute` в `catalog_compute.go`; метод-метки под префиксом `formula_txt_` автоматически покрываются дедупом через `isBaseTrainingCostMethod` (`catalog_dedup.go`) — сохранять префикс в новых метках.
- Diffusion-fixture уже есть в `main_test.go` (`org/stable-diffusion-qwen-caption`, ~333); тесты оценок ~201–470.

**Steps**

Фаза A — схема профиля (**ЗАКРЫТА**: применён и верифицирован green)
A1. В типе `computeProfile` (`catalog_config.go`) два опциональных поля бюджета (изображения/параметр, латентная длина); профиль diffusion-enabled при обоих >0. Расширить проверку валидности в `compileSelections`. — статус: build exit 0; статические проверки по обоим файлам без ошибок; полный прогон тестов ранее green (112/112). Замечание к чистке: строки полей вставлены без табов-отступа — прогнать `gofmt -w catalog_config.go` при следующем удобном моменте.

Фаза B — функция оценки (*зависит от A*)
B1. В `catalog_compute.go` оценка для diffusion-base: `6 × N_eff × (N_total × бюджет)`, бюджет = (изображения/параметр) × латентная длина; N_eff по reported active-параметрам при наличии, иначе total. Методы под префиксом `formula_txt_`; явные assumed-флаги и Source со ссылкой на профиль.

Фаза C — гейтинг в записи (*зависит от B*)
C1. Детекция diffusion-цели (pipeline_tag ∈ {text-to-image, image-to-image, text-to-video}; консервативные запасные сигналы через существующие паттерны `nonTextModelTagRE`, без новых регулярок). В base-ветке `modelToRecord`: для diffusion-enabled профилей оценка выдаётся даже без scratch-claim; приоритет reported compute оформляется явно. Не-diffusion модели — поведение неизменно.

Фаза D — сценарии и документация (*зависит от B–C*; параллельно с E)
D1. `catalog.scenarios.json`: два профиля-сценария (консервативный/верхний уровень бюджета, имена и описания по образцу MoE-sensitivity) + один diffusion-selection (pipeline_tags text-to-image/image-to-image/text-to-video, model_kinds base), ссылающийся на оба. Обновить `CATALOG.md`, `HOW-TO-RU.md` (новые поля профилей), методологию: assumption-тиры пересчитываются отдельно и не входят в headline SAM без явного включения.

Фаза E — тесты (*зависит от B–C*; параллельно с D)
E1. `main_test.go`: юнит-тест арифметики оценки с ожидаемым числом (по образцу ~470); гейтинг diffusion-базы + assumed-флаги; приоритет reported compute; инвариантность не-diffusion моделей; дедупликация идентичных diffusion-карточек по-прежнему схлопывается.

**Relevant files**
- `/home/bekket/go/src/hfScraper/catalog_compute.go` — функция оценки (B1)
- `/home/bekket/go/src/hfScraper/catalog_record.go` — гейтинг base-ветки `modelToRecord` (C1)
- `/home/bekket/go/src/hfScraper/catalog.scenarios.json` — новые профили и selection (D1)
- `/home/bekket/go/src/hfScraper/main_test.go` — diffusion-fixture уже присутствуют (~333), тесты оценок ~201–470 (E1)
- `/home/bekket/go/src/hfScraper/CATALOG.md`, `HOW-TO-RU.md`, `METHODOLOGY-2025-RU.md` — документация (D1)

**Verification**
1. Целевые юнит-тесты Фазы E + повторный полный прогон: существующие тесты (~201–470) остаются без изменений; `go build ./... && go vet ./...` green после каждого шага B/C.
2. Ручной прогон diffusion-selection на локальном fixture/эндпоинте: оба бакета методов и assumed-флаги видны в выводе; не-diffusion selections дают неизменённые результаты (проверка инвариантности).

**Decisions**
- Move-only разбивка выполнена и верифицирована ранее; из плана исключена.
- Механизм diffusion-затрат: два именованных профиля-сценария вместо встроенного бюджета; значения допущений живут только в конфиге, код несёт лишь арифметику.
- Консервативный тир — документированный класс бюджетов SD/SDXL (~O(1)–O(10) изображений/параметр × латентная длина ~10³); верхний тир — 10³×10³ (верхняя граница). Assumption-тиры исключаются из headline SAM без явного включения.
- Исключено: новые регулярки извлечения image-budget из карточек (reported-парсер уже покрывает), изменения scratch-классификатора и дедупликации, audio/music/3d pipelines.

**Further Considerations**
1. Точные значения консервативного тира настраиваются в одном профиле без кода; при необходимости подстроить под конкретный парк моделей.
