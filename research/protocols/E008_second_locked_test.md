# E008 — вторая закрытая оценка

Дата регистрации: 23 августа 2026 года. Протокол фиксируется до построения нового holdout и до evaluation checkpoints на нём.

## Основание

E006/E007 являются post-test validation ablations. Они показали, что random-depth support 8…24 переносит лучший readout на T=16, а final-only режим лучше auxiliary на трёх из четырёх training seed. Старый test E004 нельзя использовать для выбора новой модели, поэтому E008 строит второй непересекающийся по документам holdout.

## Выбранный метод

Новый candidate фиксируется как `combined_random8_24_final` при inference T=16. Основное matched-seed сравнение использует seed 23 и 47:

| Роль | Recipe | Seed | T | Checkpoint SHA-256 |
| --- | --- | ---: | ---: | --- |
| candidate | random8…24 final-only | 23 | 16 | `efa04970f75855f120eed85764204d0c62cafd1c568fd626158daea0fb574d2e` |
| candidate | random8…24 final-only | 47 | 16 | `6eee54ff7c8b64fb1e6980decb9731bd75e6e8ecd242fa33daf691400a42ff69` |
| previous | random8…16 auxiliary | 23 | 12 | `4b5b28c38b6de2e4a6b670f6a7ce8798e3a4593780b890586e0f333533968c2e` |
| previous | random8…16 auxiliary | 47 | 12 | `c5fac83cc147c5abd91e49b818620f3162cd7d38a69ae35eb79eeb06ad02f146` |

Для оценки training variance дополнительно фиксируются final-only seed 71 (`d9ed2d4c7ec5903ae883d926c79f750e44cc11311a5c4cbb9cf5c5f956af7e94`) и seed 89 (`ada8510904b6460c22373a369ec4dd82dd0b980f277ce22dbc6a218f8f29a095`). Они не участвуют в matched comparison со старым методом.

## Новый holdout

Источник, revision, tokenizer и split rule совпадают с E004. Из test split полностью исключаются все 1,251 документа, использованные в E004. Feasibility-проход до построения holdout показал, что после документного исключения доступны ровно 1,456 непересекающихся окон, поэтому используются все они. Окна имеют 513 токенов и дают 745,472 target-токена. Содержимое и метрики не просматриваются до фиксации NPZ hash и evaluation-команды.

Первоначально были зарегистрированы 2,048 окон. Этот объём оказался невозможен при полном исключении документов E004; изменение до максимальных 1,456 сделано после сообщения `eligible_windows=1456`, до создания NPZ и до любой evaluation модели.

Построенный holdout содержит 1,080 документов. SHA-256 NPZ: `df609e39b23085c75d4c33bf1a061f2c48a99023f6708daf3e475ba8d1e30b79`. Пересечение множества document hashes с E004 равно нулю.

## Evaluation

Все шесть checkpoints оцениваются на одних окнах при T={12,16,24,32}. Основные решения:

**Q1, качество нового recipe.** На T=16 candidate улучшает NLL относительно прежнего checkpoint при его выбранном T=12 на обоих matched seed, а среднее улучшение двух seed не меньше 0.01.

**Q2, matched compute.** На T=16 candidate улучшает NLL относительно прежнего checkpoint при T=16 на обоих seed.

**Q3, стабильность диапазона.** Для всех четырёх candidate seed `NLL@24−NLL@16≤0.02`.

Paired document bootstrap используется для Q1/Q2 отдельно по seed. Решение определяется training-seed повторяемостью; bootstrap не подменяет training seed.

## Ограничение

E008 является последним открытием test в проекте. После него recipe, checkpoint set, inference depth и пороги не меняются. Независимо от результата дополнительный подбор на test не проводится.
