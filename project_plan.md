# План исследования

Дата публикационного среза: 23 августа 2026 года.

## Вопрос

Можно ли обучить looped Transformer так, чтобы дополнительные применения общего блока продолжали улучшать next-token prediction, а не только переносили оптимум на другую фиксированную глубину?

Разделяем два критерия: минимальный validation NLL по глубине и изменение NLL при увеличении T для одного checkpoint.

## Архитектура

```text
tokens → tied embedding → RMSNorm
       → repeated shared core
       → RMSNorm → tied LM head
```

Core содержит два Qwen-style decoder blocks с GQA, QK-norm, RoPE и SwiGLU. Проверяются relative input injection, Depth-RoPE, fixed/random training depth, промежуточный LM loss и порядок Attention/MLP.

Отдельные полноразмерные prelude/coda не используются: при width 512 они выводят модель за лимит 10M параметров.

## Последовательность экспериментов

| ID | Назначение | Решение |
| --- | --- | --- |
| E001 | Разделить injection, Depth-RoPE и нормализацию | Выбрать механизмы для GPU-screening |
| E002 | Проверить fixed/random depth и auxiliary loss | Передать baseline и один candidate в BPE-подтверждение |
| E003 | Повторить сравнение на 9.44M BPE-модели и двух seed | Продолжать до 100M только по зарегистрированному порогу |
| E004 | Один раз проверить выбранные checkpoint и depth на test | Не использовать test для последующего подбора |
| E005 | Проверить порядок MLP→Attention в двух training recipes | Завершён: порядок ухудшил NLL на обоих seed; сохранить Attention→MLP |
| E006 | Расширить random-depth support с 8…16 до 8…24 при средней глубине 16 | Завершён: рабочая область сдвинулась, но экстраполяция после T=24 снова ухудшается |
| E007 | Повторить final-only/auxiliary сравнение E006 на seed 71/89 | Оценить training-seed variance без изменения механизма |

## Контроли

- одинаковые train windows и evaluation documents внутри сравнения;
- одинаковые tokenizer, context, optimizer, LR schedule и token budget;
- несколько training seed;
- parameter count включает embeddings;
- один checkpoint оценивается на всей сетке глубин;
- test открывается после выбора метода, checkpoint и inference depth;
- bootstrap по документам не подменяет вариативность training seed.

## Правило масштабирования

К 100M продолжаются только candidate и matched control, если эффект повторяется на обоих seed и не ухудшает позднюю depth curve сверх зарегистрированного допуска. Изменение метода после просмотра результата получает новый ID и новый протокол.

## Артефакты запуска

Сохраняются аргументы и hashes исходников, seed и cursor данных, learning curve, optimizer/scaler/RNG checkpoint, число токенов, память и время, depth sweep, per-document losses и решение по заранее заданному правилу.

## Завершение

Финальный комплект содержит код обучения и evaluation, отчёт обо всех исходах, tokenizer, config, выбранные веса, model card и контрольные суммы. Тяжёлые train-данные и optimizer checkpoints публикуются отдельно либо воспроизводятся по закреплённому manifest.
