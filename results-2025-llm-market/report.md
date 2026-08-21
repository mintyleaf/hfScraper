# Оценка стоимости обучения моделей Hugging Face

Снимок сформирован: 2026-08-21T01:07:20.217210526Z

## public_text_llm_2025

Public 2025 text LLM repositories with weights; confirmed independent base runs use formula.txt and derivatives receive cost only when their own compute/cost is reported

- Итоговая стоимость: **$50545833.34**
- Сценарий строго 20 токенов/параметр, MoE по active × total: **$20571760.43**
- Буквальный upper-сценарий 20 токенов/параметр и total² для MoE: **$271571788.26**
- Репозиториев в целевой выборке: 447533
- Самостоятельных training run с рассчитанной стоимостью: 1094
- Уникальных владельцев: 64688
- Скачиваний: 309184434; likes: 331594

| Категория | Репозиториев | Run с cost | Стоимость, USD |
|---|---:|---:|---:|
| base | 94129 | 545 | 48699290.19 |
| fork | 28895 | 0 | 0.00 |
| finetune | 142304 | 334 | 1841879.74 |
| adapter | 51504 | 215 | 4663.41 |
| quantized | 119382 | 0 | 0.00 |
| merge | 11319 | 0 | 0.00 |

### Base по числу параметров

| Диапазон | Base-репозиториев | Подтверждённых run с cost | Стоимость, USD |
|---|---:|---:|---:|
| <2B | 47335 | 418 | 1273803.37 |
| 2-<7B | 10840 | 47 | 3025085.85 |
| 7-<13B | 11495 | 37 | 8348937.65 |
| 13-<34B | 5279 | 23 | 11716336.66 |
| 34-<70B | 75 | 8 | 2270757.61 |
| 70-<120B | 371 | 6 | 11110221.92 |
| 120-<500B | 94 | 4 | 4418596.43 |
| >=500B | 65 | 2 | 6535550.71 |
| unknown | 18575 | 0 | 0.00 |

Base-кандидатов на анализ карточки: 94151; признаки независимого pretraining/base найдены у 3958; без достаточного evidence: 90193. Fine-tune/adapter-кандидатов на compute: 287952; найден compute/cost: 699; без данных: 287253. Репозиториев с весами после первичного фильтра: 902093.

Для base-модели используется FLOPs = 6 × N × T из `formula.txt`. Если карточка сообщает фактический объём pretraining-токенов, берётся он; иначе T = 20 × N. Для dense это сводится к cost = 155.881361644759 × P² USD. Для MoE при наличии публичного active count FLOPs/token считаются по active-параметрам, а token fallback — по total. Fork, quantization, conversion и merge не получают стоимость; fine-tune/adapters входят только при опубликованном compute/cost.

Ограничения интерпретации:

- Первичный отбор 2025 сделан по `createdAt` репозитория.
- Если в training section явно указан другой год и не указан 2025, такой run исключён; без даты используется год `createdAt` как допущение.
- Base без достаточного independent-pretraining evidence и derivatives без опубликованного compute в сумму не входят.
- Копии с идентичной карточкой дедуплицируются; переписанные зеркала без общего run ID всё ещё невозможно надёжно связать.
- Один и тот же текст карточки мог использоваться для разных запусков; консервативная дедупликация может занижать нижнюю границу.
- Формула оценивает H100 GPU-rental equivalent; CPU, сеть, storage, данные и работа команды не включены.
- Для MoE основной estimate использует опубликованное число active-параметров; `summary.json` отдельно сохраняет буквальный total² upper-сценарий.
- Приватные, недоступные и уже удалённые репозитории не входят в текущий публичный снимок.

## Владельцы с наибольшей оценкой

| Владелец | Тип | Моделей с cost | Стоимость, USD |
|---|---:|---:|---:|
| [swiss-ai](https://huggingface.co/swiss-ai) | unknown | 2 | 9195426.25 |
| [Qwen](https://huggingface.co/Qwen) | unknown | 6 | 9146595.16 |
| [moonshotai](https://huggingface.co/moonshotai) | unknown | 2 | 3888829.70 |
| [allenai](https://huggingface.co/allenai) | unknown | 4 | 3272783.16 |
| [skt](https://huggingface.co/skt) | unknown | 2 | 3237170.57 |
| [ibm-granite](https://huggingface.co/ibm-granite) | unknown | 12 | 3093269.06 |
| [nvidia](https://huggingface.co/nvidia) | unknown | 12 | 3023673.12 |
| [baidu](https://huggingface.co/baidu) | unknown | 4 | 2214700.48 |
| [zai-org](https://huggingface.co/zai-org) | unknown | 2 | 1994102.22 |
| [upstage](https://huggingface.co/upstage) | unknown | 1 | 1842517.69 |
| [XiaomiMiMo](https://huggingface.co/XiaomiMiMo) | unknown | 1 | 1526353.18 |
| [OpenTrouter](https://huggingface.co/OpenTrouter) | unknown | 1 | 1022976.00 |
| [utter-project](https://huggingface.co/utter-project) | unknown | 1 | 705747.51 |
| [1Covenant](https://huggingface.co/1Covenant) | unknown | 1 | 623697.39 |
| [EssentialAI](https://huggingface.co/EssentialAI) | unknown | 1 | 544089.95 |
| [AstroMLab](https://huggingface.co/AstroMLab) | unknown | 1 | 434750.00 |
| [Kwai-Klear](https://huggingface.co/Kwai-Klear) | unknown | 1 | 428673.74 |
| [inclusionAI](https://huggingface.co/inclusionAI) | unknown | 6 | 403872.72 |
| [dots-studio](https://huggingface.co/dots-studio) | unknown | 1 | 311582.11 |
| [arcee-ai](https://huggingface.co/arcee-ai) | unknown | 3 | 301188.63 |
| [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) | unknown | 1 | 268436.31 |
| [IQuestLab](https://huggingface.co/IQuestLab) | unknown | 2 | 255885.47 |
| [PsycheFoundation](https://huggingface.co/PsycheFoundation) | unknown | 1 | 251937.39 |
| [ByteDance](https://huggingface.co/ByteDance) | unknown | 2 | 246216.41 |
| [ByteDance-Seed](https://huggingface.co/ByteDance-Seed) | unknown | 3 | 239883.68 |
| [mistralai](https://huggingface.co/mistralai) | unknown | 5 | 222018.10 |
| [tiiuae](https://huggingface.co/tiiuae) | unknown | 5 | 187364.57 |
| [Infinigence](https://huggingface.co/Infinigence) | unknown | 1 | 187057.63 |
| [marin-community](https://huggingface.co/marin-community) | unknown | 2 | 174900.77 |
| [pfnet](https://huggingface.co/pfnet) | unknown | 3 | 168473.98 |
| [CYFRAGOVPL](https://huggingface.co/CYFRAGOVPL) | unknown | 4 | 160793.04 |
| [microsoft](https://huggingface.co/microsoft) | unknown | 6 | 146657.89 |
| [Kirim-ai](https://huggingface.co/Kirim-ai) | unknown | 1 | 119347.20 |
| [common-pile](https://huggingface.co/common-pile) | unknown | 1 | 109158.37 |
| [omunaman](https://huggingface.co/omunaman) | unknown | 1 | 68186.73 |
| [Pinkstack](https://huggingface.co/Pinkstack) | unknown | 1 | 40583.55 |
| [VillanovaAI](https://huggingface.co/VillanovaAI) | unknown | 1 | 40366.45 |
| [HiTZ](https://huggingface.co/HiTZ) | unknown | 2 | 33931.07 |
| [open-thoughts](https://huggingface.co/open-thoughts) | unknown | 1 | 28860.00 |
| [Motif-Technologies](https://huggingface.co/Motif-Technologies) | unknown | 2 | 26208.90 |
| [kakaocorp](https://huggingface.co/kakaocorp) | unknown | 3 | 25073.99 |
| [speakleash](https://huggingface.co/speakleash) | unknown | 1 | 19444.96 |
| [KORMo-Team](https://huggingface.co/KORMo-Team) | unknown | 1 | 18036.25 |
| [toksuite](https://huggingface.co/toksuite) | unknown | 12 | 17903.40 |
| [dvruette](https://huggingface.co/dvruette) | unknown | 3 | 17819.49 |
| [BSC-LT](https://huggingface.co/BSC-LT) | unknown | 1 | 14638.01 |
| [prithivMLmods](https://huggingface.co/prithivMLmods) | unknown | 1 | 13313.69 |
| [Zyphra](https://huggingface.co/Zyphra) | unknown | 1 | 12182.79 |
| [DMIR01](https://huggingface.co/DMIR01) | unknown | 6 | 11526.13 |
| [Darwinlabsai](https://huggingface.co/Darwinlabsai) | unknown | 1 | 10822.50 |
| [ilsp](https://huggingface.co/ilsp) | unknown | 1 | 10487.16 |
| [openbmb](https://huggingface.co/openbmb) | unknown | 2 | 10473.14 |
| [LumiOpen](https://huggingface.co/LumiOpen) | unknown | 1 | 10052.02 |
| [Paschalidis-NOC-Lab](https://huggingface.co/Paschalidis-NOC-Lab) | unknown | 1 | 10052.02 |
| [GSAI-ML](https://huggingface.co/GSAI-ML) | unknown | 1 | 10015.31 |
| [Dream-org](https://huggingface.co/Dream-org) | unknown | 1 | 9040.75 |
| [HPLT](https://huggingface.co/HPLT) | unknown | 9 | 8708.70 |
| [nc-ai-consortium](https://huggingface.co/nc-ai-consortium) | unknown | 2 | 8076.73 |
| [raxcore-dev](https://huggingface.co/raxcore-dev) | unknown | 1 | 7992.00 |
| [WeiboAI](https://huggingface.co/WeiboAI) | unknown | 1 | 7800.00 |
| [geodesic-research](https://huggingface.co/geodesic-research) | unknown | 1 | 7327.70 |
| [kz919](https://huggingface.co/kz919) | unknown | 8 | 7165.29 |
| [Ak015](https://huggingface.co/Ak015) | unknown | 1 | 6891.98 |
| [vngrs-ai](https://huggingface.co/vngrs-ai) | unknown | 1 | 5553.60 |
| [TroyDoesAI](https://huggingface.co/TroyDoesAI) | unknown | 1 | 4856.40 |
| [TurkuNLP](https://huggingface.co/TurkuNLP) | unknown | 7 | 4343.35 |
| [Sheikh-F1](https://huggingface.co/Sheikh-F1) | unknown | 1 | 4262.40 |
| [ServiceNow-AI](https://huggingface.co/ServiceNow-AI) | unknown | 1 | 3639.66 |
| [iamshnoo](https://huggingface.co/iamshnoo) | unknown | 20 | 3520.49 |
| [ai-sage](https://huggingface.co/ai-sage) | unknown | 1 | 3221.06 |
| [Polygl0t](https://huggingface.co/Polygl0t) | unknown | 5 | 3172.41 |
| [deepvk](https://huggingface.co/deepvk) | unknown | 2 | 2872.03 |
| [DMindAI](https://huggingface.co/DMindAI) | unknown | 2 | 2664.00 |
| [JetBrains](https://huggingface.co/JetBrains) | unknown | 1 | 2518.16 |
| [sbordt](https://huggingface.co/sbordt) | unknown | 1 | 2430.44 |
| [Nanbeige](https://huggingface.co/Nanbeige) | unknown | 1 | 2412.03 |
| [barqahmed01](https://huggingface.co/barqahmed01) | unknown | 1 | 2293.80 |
| [novelcore](https://huggingface.co/novelcore) | unknown | 3 | 2171.41 |
| [cjvt](https://huggingface.co/cjvt) | unknown | 2 | 1857.40 |
| [PokeeAI](https://huggingface.co/PokeeAI) | unknown | 1 | 1776.00 |
| [amd](https://huggingface.co/amd) | unknown | 1 | 1510.30 |
| [abdoelsayed](https://huggingface.co/abdoelsayed) | unknown | 10 | 1443.00 |
| [Benyucong](https://huggingface.co/Benyucong) | unknown | 1 | 1420.80 |
| [boun-tabilab](https://huggingface.co/boun-tabilab) | unknown | 1 | 1165.27 |
| [AI4PH](https://huggingface.co/AI4PH) | unknown | 3 | 1095.20 |
| [WiroAI](https://huggingface.co/WiroAI) | unknown | 1 | 1065.60 |
| [amirakhlaghiqqq](https://huggingface.co/amirakhlaghiqqq) | unknown | 1 | 1065.60 |
| [almanach](https://huggingface.co/almanach) | unknown | 1 | 1060.94 |
| [FractalAIResearch](https://huggingface.co/FractalAIResearch) | unknown | 1 | 967.00 |
| [aisingapore](https://huggingface.co/aisingapore) | unknown | 2 | 791.80 |
| [aaronmoo12](https://huggingface.co/aaronmoo12) | unknown | 1 | 754.80 |
| [saiteja33](https://huggingface.co/saiteja33) | unknown | 1 | 740.00 |
| [ik-ram28](https://huggingface.co/ik-ram28) | unknown | 1 | 710.40 |
| [facebook](https://huggingface.co/facebook) | unknown | 5 | 702.32 |
| [AI71ai](https://huggingface.co/AI71ai) | unknown | 2 | 666.00 |
| [Salesforce](https://huggingface.co/Salesforce) | unknown | 1 | 643.47 |
| [RefinedNeuro](https://huggingface.co/RefinedNeuro) | unknown | 1 | 621.60 |
| [manelalab](https://huggingface.co/manelalab) | unknown | 1 | 536.55 |
| [qvac](https://huggingface.co/qvac) | unknown | 1 | 536.41 |
| [mist-models](https://huggingface.co/mist-models) | unknown | 2 | 502.31 |
