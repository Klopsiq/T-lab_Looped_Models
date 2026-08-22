# E004 — закрытый test

E004 зарегистрирован после выбора метода, checkpoint и inference depth. После первого вычисления test loss результат не использовался для нового выбора или обучения.

## Результат

| Checkpoint | Depth | Validation NLL | Test NLL | Test PPL |
| --- | ---: | ---: | ---: | ---: |
| `combined_random8_16_aux`, seed 47 | **12** | **4.2058** | **4.2103** | **67.38** |
| тот же candidate | 16 | 4.2115 | 4.2165 | 67.79 |
| `relative_fixed16`, seed 47 | 16 | 4.2340 | 4.2418 | 69.53 |

При общей глубине T=16 candidate улучшает test NLL на **0.0253** и PPL на 1.74 относительно matched baseline. На validation соответствующий выигрыш был 0.0226 NLL. Выбранная рабочая точка candidate T=12 улучшает test NLL относительно рабочей точки baseline T=16 на 0.0315.

Абсолютный разрыв validation→test мал и имеет одинаковый знак: +0.0045 NLL для candidate T=12, +0.0050 для candidate T=16 и +0.0077 для baseline. Это согласуется с переносом эффекта на закрытые документы, но один test shard не является доказательством переноса на другие корпуса или масштабы.

## Что подтверждено

- Положительный эффект candidate в пределах training-depth повторился на двух validation seed и matched test сравнении.
- Выбор T=12 по validation не развалился на test: это также лучшая из двух заранее разрешённых test-точек candidate.
- E004 не проверяет экстраполяцию T>16. Полный test sweep был заранее запрещён, поэтому отрицательный вывод о поздней глубине остаётся основан на validation E003.
- Обучение остановлено на 24,969,216 токенах по зарегистрированному правилу. Test не используется как повод продолжить до 100M.

## Проверяемость

- [Предварительная регистрация](E004_registration.json).
- [Протокол](../research/protocols/E004_locked_test.md).
- Test windows: SHA-256 `0d91ce72d1e2f79d88a5656a1c5ffbe8321b4071494b37951f427dec130199fb`, 1,048,576 target-токенов из 1,251 документа.
- [Candidate result](E004_candidate_test.json), SHA-256 `decbff1cfb4506f812d29f81d448f652eda4e1d804cf2ae95fce4f0c9ea8c26d`.
- [Baseline result](E004_baseline_test.json), SHA-256 `7fd82aa9a2450b0b695581cd22e22bcdf74a531b440a3b2740743530c7be2a64`.
- Checkpoint hashes внутри обоих JSON совпадают с выбранными до test.

Повторный test-eval после этой записи не планируется.
