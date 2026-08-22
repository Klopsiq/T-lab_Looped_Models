# Looped Models

Исследование полезной рекуррентной глубины небольшого causal Transformer. Один и тот же блок применяется несколько раз, поэтому вычислительную глубину можно менять без увеличения числа параметров.

**Срез публикации: 23 августа 2026 года.**

## Основной результат

Финальный checkpoint проекта — **E003 `combined_random8_16_aux`, seed 47, inference depth T=12**. Он был выбран по validation до единственного открытия locked test. Модель с random-depth training и затухающим промежуточным LM loss улучшает качество внутри обучаемого диапазона `T=8…16`, но не сохраняет пользу дополнительных циклов после `T=16`.

Дополнительная проверка порядка residual sublayers показала, что `MLP→Attention` ухудшает NLL на обоих seed. Поэтому финальная архитектура сохраняет порядок `Attention→MLP`.

Post-test абляция с расширением random-depth support до `T=24` перенесла лучший readout с T=12 на T=16 и уменьшила деградацию `16→24` примерно в четыре-пять раз. После новой границы качество снова ухудшается, поэтому эффект интерпретируется как сдвиг рабочей области, а не свободное test-time scaling.

Репликация на четырёх training seed показала, что auxiliary loss помогает только одному seed. Final-only оказался лучше на остальных трёх и поэтому является наиболее перспективным продолжением, но validation-only результаты E005–E007 не меняют финальный checkpoint задним числом.

На двух seed выигрыш candidate относительно fixed-depth baseline при одинаковом `T=16` составил `0.0292` и `0.0226 NLL`. На закрытом test:

| Модель | Глубина | NLL | PPL |
| --- | ---: | ---: | ---: |
| random depth + auxiliary | 12 | **4.2103** | **67.38** |
| random depth + auxiliary | 16 | 4.2165 | 67.79 |
| fixed-depth baseline | 16 | 4.2418 | 69.53 |

После `T=16` candidate деградирует быстрее baseline. Поэтому улучшение лучшего readout и полезный test-time scaling рассматриваются как разные свойства. Подробный разбор: [REPORT.md](REPORT.md).

## Статус последующих экспериментов

- **E005 подтвердил**, что перестановка на `MLP→Attention` ухудшает NLL; в финальной архитектуре сохранён порядок `Attention→MLP`.
- **E006 подтвердил**, что расширение train-depth support с `8…16` до `8…24` переносит оптимальный readout с T=12 на T=16 и улучшает поведение около новой границы.
- **E007 показал на четырёх seed**, что final-only устойчивее auxiliary loss: он выиграл на трёх seed из четырёх и достиг лучшего validation NLL 4.1799.
- Эти результаты получены после E004 на validation. Они описывают направления роста, но не имеют новой locked-test оценки и потому не заменяют E003.
- **E008 не выполнялся**. Его регистрация сохранена как готовый протокол возможного продолжения, а экспериментальная часть этого среза завершена на E003/E004.

## Ограничения задания

- не более 10M уникальных параметров;
- не более 100M предъявленных обучающих токенов на одну модель;
- обучение с нуля на FineWeb;
- causal next-token prediction;
- checkpoint и воспроизводимый код обучения и evaluation.

Финальная исследованная модель содержит **9,440,513 параметров**, использует vocabulary 8192 и context 512.

## Структура

```text
looped_models/       модель, данные и CPU experiment runner
scripts/             GPU training, evaluation, анализ и экспорт
configs/             конфигурации экспериментов
research/protocols/  условия экспериментов, записанные до анализа
reports/             итоговые JSON, CSV, графики и отчёты
tests/               проверки причинности, checkpoint и resume
artifacts/           комплект выбранного checkpoint для Hugging Face
```

Тяжёлые данные, optimizer checkpoints, окружения и сырые GPU-архивы не входят в Git.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[bpe,plots]'
```

Для CUDA необходимо установить сборку PyTorch, совместимую с драйвером конкретной машины.

## Проверки

```bash
python3 -m unittest discover -s tests -v
```

Проверяются причинная маска, tying весов, лимит параметров, Depth-RoPE, readout промежуточных состояний, gradient checkpointing, сохранение весов и точный CPU resume.

## Подготовка данных

```bash
python3 scripts/prepare_bpe_data.py --help
```

Разбиение документов и exact dedup выполняются до обучения tokenizer. Tokenizer обучается только на train. Точный manifest корпуса описан в [DATA_BPE8K.md](research/DATA_BPE8K.md).

## Обучение

```bash
python3 scripts/e003_bpe_train.py \
  --arms relative_fixed16 combined_random8_16_aux \
  --seeds 23 47 \
  --target-tokens 25000000 \
  --schedule-tokens 100000000 \
  --lr 0.002
```

Эксперимент с порядком MLP→Attention:

```bash
python3 scripts/e005_order_interaction.py \
  --arms relative_fixed16_mlp_first combined_random8_16_aux_mlp_first \
  --seeds 23 47 \
  --target-tokens 25000000 \
  --schedule-tokens 100000000 \
  --lr 0.002
```

Runner регистрирует конфигурацию и hashes исходников до обучения, сохраняет checkpoints и поддерживает resume.

## Результаты

- [E001: малый факторный pilot](reports/E001_report.md)
- [E002: screening механизмов](reports/E002_report.md)
- [E003: подтверждение на FineWeb BPE8k](reports/E003_report.md)
- [E004: закрытый test](reports/E004_report.md)
- [E005: порядок MLP→Attention](reports/E005_report.md)
- [E006: диапазон обучающих глубин](reports/E006_report.md)
- [E007: репликация на новых seed](reports/E007_report.md)
- [Журнал исследовательских решений](research/JOURNAL.md)
- [План исследования](project_plan.md)

## Публикация

- [Финальный checkpoint E003 на Hugging Face](https://huggingface.co/Klopsiq/looped-models-bpe8k-final)
- [Model card и состав публикуемого bundle](artifacts/looped-models-bpe8k-final/README.md)

В публикацию входят веса, tokenizer, конфигурация, пример inference и результаты locked test.
