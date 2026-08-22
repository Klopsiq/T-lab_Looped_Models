# FineWeb BPE8k data card

Этот корпус подготовлен для E003 и финального этапа задания. Бинарные token files не предназначены для Git; `manifest.json`, tokenizer и выбранный checkpoint должны публиковаться вместе с финальным комплектом.

## Источник

- Dataset: `HuggingFaceFW/fineweb`, config `sample-10BT`.
- Parquet: `sample/10BT/000_00000.parquet`.
- Revision: `9bb295ddab0e05d785b879661af7260fed5140fc`.
- Размер скачанного файла: 2,147,292,183 байта.
- SHA-256: `6b552ea48424648dc86d00df276f93fdfc55e9ad342ce3e4affc23a3a370792b`.

Исходный файл доступен на [Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/9bb295ddab0e05d785b879661af7260fed5140fc/sample/10BT/000_00000.parquet). Повторная подготовка обязана проверить SHA-256 до чтения данных.

## Защита от leakage

Для каждого документа вычисляется SHA-256 от полного текста после нормализации whitespace. Exact-дубликаты с одинаковым хешем удаляются до split. Остаток `int(hash[:16], 16) mod 100` задаёт split: 0…89 train, 90…94 validation, 95…99 test.

Tokenizer обучается только на первых 50,000 уникальных документах, уже назначенных в train. Validation и test не используются для построения vocabulary. Близкие смысловые дубликаты не удаляются, поэтому отсутствие near-duplicate leakage не доказано.

## Tokenizer и packing

- Hugging Face `tokenizers`, byte-level BPE.
- Vocabulary 8192, `<|endoftext|>` имеет ID 0.
- Byte alphabet включён целиком; неизвестные UTF-8 bytes отсутствуют.
- Документы разделяются EOS, IDs хранятся little-endian `uint16`.
- Train windows длины 513 берутся с шагом 512 и могут пересекать EOS; 512 target-токенов считаются предъявленными модели.
- Для validation отдельно сохранены окна, не пересекающие границы документа. Это позволяет paired bootstrap по документам.

Проверен точный round-trip на ASCII, пробелах, переводах строк и многоязычном UTF-8 тексте.

## Объём

| Split | Документы | Токены | UTF-8 bytes |
| --- | ---: | ---: | ---: |
| Train | 133,552 | 110,000,343 | 413,775,189 |
| Validation | 4,953 | 4,000,762 | 15,077,195 |
| Test, закрыт | 4,957 | 4,001,038 | 14,988,715 |

Всего при подготовке просмотрено 148,427 уникальных документов; удалены 4 exact-дубликата. Train содержит 214,844 непересекающихся окон при context 512. Траектории 100M требуют 195,312 окон, поэтому loader не повторяет данные в пределах бюджета.

Основной validation eval: 2,048 детерминированно выбранных окон, 1,048,576 target-токенов, 1,258 документа. Приоритет окна задаётся SHA-256 от `document_hash:start`, поэтому выбор не зависит от порядка обхода внутри split.

## Файлы

| Файл | Байты | SHA-256 |
| --- | ---: | --- |
| `train.bin` | 220,000,686 | `acf0c79f3ea1dff58b9e113ff46abac8503de8584e9ad462c04a185b4b42c9ad` |
| `validation.bin` | 8,001,524 | `54b6109a6a7a60625487c7ef75ff96439f12085740bc2bfd3e75eeb4c0b6737a` |
| `test.bin` | 8,002,076 | `b534ac144cbef346b398f745ed2dcbecc1b2c3973157a2f9bc49e55c8ff46ef4` |
| `validation_windows.npz` | 1,541,929 | `71bdbd39cceb57339162cb6135ef27e7768d6736a2a5ee5b2d676108bc1f6b64` |
| `tokenizer.json` | 546,511 | `59907441ee1b0d0f20b10571da1b5c58928d9706a788f23f1fe2bc7c2fba8c1c` |

## Воспроизведение

```bash
PYTHONPATH=.bpe-deps python3 scripts/prepare_bpe_data.py \
  --source data/source/fineweb_sample10BT_000_00000.parquet \
  --source-url https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/9bb295ddab0e05d785b879661af7260fed5140fc/sample/10BT/000_00000.parquet \
  --source-revision 9bb295ddab0e05d785b879661af7260fed5140fc \
  --source-sha256 6b552ea48424648dc86d00df276f93fdfc55e9ad342ce3e4affc23a3a370792b \
  --output data/fineweb_bpe8k_v1
```
