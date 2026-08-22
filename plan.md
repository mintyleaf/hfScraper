## Plan: Diffusion-затраты через сценарии-профили 

**TL;DR.** Сначала — diffusion-затраты как два именованных compute-профиля-сценария (консервативный/верхний уровень бюджета) в `catalog.scenarios.json` по образцу MoE-sensitivity; reported compute сохраняет приоритет, scratch-классификатор и дедупликация не трогаются. Scope согласован: text2img + img2img + text-to-video; assumption-оценки только с явными флагами и отдельным бакетом методов.


**Steps**



Фаза A — схема профиля 
A1. В типе `computeProfile` (`catalog_config.go`) два опциональных поля бюджета (изображения/параметр, латентная длина); профиль diffusion-enabled при обоих >0. Расширить проверку валидности в `compileSelections`.

Фаза B — функция оценки (*зависит от A*)
B1. В `catalog_compute.go` оценка для diffusion-base: `6 × N_eff × (N_total × бюджет)`, бюджет = (изображения/параметр) × латентная длина; N_eff по reported active-параметрам при наличии, иначе total. Методы под префиксом `formula_txt_`; явные assumed-флаги и Source со ссылкой на профиль.

Фаза C — гейтинг в записи (*зависит от B*)
C1. Детекция diffusion-цели (pipeline_tag ∈ {text-to-image, image-to-image, text-to-video}; консервативные запасные сигналы через существующие паттерны `nonTextModelTagRE`, без новых регулярок). В base-ветке `modelToRecord`: для diffusion-enabled профилей оценка выдаётся даже без scratch-claim; приоритет reported compute оформляется явно. Не-diffusion модели — поведение неизменно.

Фаза D — сценарии и документация (*зависит от A–C*; параллельно с E)
D1. `catalog.scenarios.json`: два профиля-сценария (консервативный/верхний уровень бюджета, имена и описания по образцу MoE-sensitivity) + один diffusion-selection (pipeline_tags text-to-image/image-to-image/text-to-video, model_kinds base), ссылающийся на оба. Обновить `CATALOG.md`, `HOW-TO-RU.md` (новые поля профилей), методологию: assumption-тиры пересчитываются отдельно и не входят в headline SAM без явного включения.

Фаза E — тесты (*зависит от A–C*; параллельно с D)
E1. `main_test.go`: юнит-тест арифметики оценки с ожидаемым числом (по образцу ~470); гейтинг diffusion-базы + assumed-флаги; приоритет reported compute; инвариантность не-diffusion моделей; дедупликация идентичных diffusion-карточек по-прежнему схлопывается.

**Relevant files**
- Фаза 0: новый набор файлов пакета `main` в `/home/bekket/go/src/hfScraper/` (см. разбивку выше); исходные декларации покидают `catalog.go`.
- `/home/bekket/go/src/hfScraper/catalog.scenarios.json` — новые профили и selection
- `/home/bekket/go/src/hfScraper/main_test.go` — diffusion-fixture уже присутствуют (~333), тесты оценок ~201–470
- `/home/bekket/go/src/hfScraper/CATALOG.md`, `HOW-TO-RU.md`, `METHODOLOGY-2025-RU.md` — документация

**Verification**
1. Целевые юнит-тесты Фазы E + повторный полный прогон: существующие тесты (~201–470) остаются без изменений.
2. Ручной прогон diffusion-selection на локальном fixture/эндпоинте: оба бакета методов и assumed-флаги видны в выводе; не-diffusion selections дают неизменённые результаты (проверка инвариантности).

**Decisions**
- Механизм diffusion-затрат: два именованных профиля-сценария вместо встроенного бюджета; значения допущений живут только в конфиге, код несёт лишь арифметику.
- Консервативный тир — документированный класс бюджетов SD/SDXL (~O(1)–O(10) изображений/параметр × латентная длина ~10³); верхний тир — 10³×10³ (верхняя граница). Assumption-тиры исключаются из headline SAM без явного включения.
- Исключено: новые регулярки извлечения image-budget из карточек (reported-парсер уже покрывает), изменения scratch-классификатора и дедупликации, audio/music/3d pipelines.

**Further Considerations**
1. Точные значения консервативного тира настраиваются в одном профиле без кода; при необходимости подстроить под конкретный парк моделей.
