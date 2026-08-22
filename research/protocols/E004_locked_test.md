# E004 — одноразовая оценка на закрытом test

Статус: зарегистрирован после всех решений по E003 и до первого вычисления test loss.

## Зафиксированный выбор

Основной checkpoint: `combined_random8_16_aux`, seed 47, 24,969,216 train-токенов, SHA-256 `c5fac83cc147c5abd91e49b818620f3162cd7d38a69ae35eb79eeb06ad02f146`. Основная inference depth T=12, потому что это validation-минимум candidate на обоих подтверждающих seed.

Matched control: `relative_fixed16`, seed 47, тот же token budget, SHA-256 `446d610faef460904909cf19989843ad9e38904b57e08312dbb4982ce5687bc9`, inference depth T=16.

Seed 47 выбран для публикуемого checkpoint до test, поскольку он имеет меньший validation NLL среди двух candidate checkpoints. После E004 веса, метод, seed и depth не меняются.

## Test set

Из закрытого split тем же детерминированным правилом, что для validation, построены 2,048 within-document окон: 1,048,576 target-токенов из 1,251 документа. Ни текст, ни test loss при выборе модели не просматривались.

- `E004_locked_test_windows.npz`: SHA-256 `0d91ce72d1e2f79d88a5656a1c5ffbe8321b4071494b37951f427dec130199fb`.
- Source parquet SHA-256: `6b552ea48424648dc86d00df276f93fdfc55e9ad342ce3e4affc23a3a370792b`.
- Tokenizer SHA-256: `59907441ee1b0d0f20b10571da1b5c58928d9706a788f23f1fe2bc7c2fba8c1c`.
- Window builder SHA-256: `7016d79ed522906fd747b93d682bb83679c621d13a2546b880d62c9b52f8fa24`.
- Eval script SHA-256: `3876a0fbb5dad5eb350f67f0e9869554b9b918c4f41cf8664c2dbf9c47c38ed2`.

## Единственная оценка

В одной test-сессии вычисляются только:

1. основной checkpoint при T=12;
2. тот же checkpoint при общей глубине T=16;
3. matched baseline checkpoint при T=16.

Основной публикуемый test PPL — candidate T=12. Разность candidate−baseline при T=16 проверяет перенос парного validation-эффекта. T=12 против baseline T=16 сравнивает заранее выбранные рабочие точки каждого метода. Полный test depth sweep запрещён: он превратил бы test в дополнительный validation.

Test-результат сообщается независимо от знака. Не допускаются повторное обучение, выбор другого checkpoint/depth или ещё одна оценка на этом test после просмотра E004.
