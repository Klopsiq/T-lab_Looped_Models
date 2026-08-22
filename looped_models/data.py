"""Download a small auditable FineWeb snapshot; UTF-8 byte windows per document.

The viewer API is not a guaranteed revision-pinned export. Exact reproducibility
uses the downloaded response hashes and saved snapshot, not the observed Hub SHA.
"""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root, length=64, offsets=(0, 10000, 20000, 30000)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
    if manifest_path.exists():
        raise FileExistsError("Snapshot already prepared; use it or choose a new output directory")
    raw_dir = root/'raw'
    raw_dir.mkdir(exist_ok=True)
    pages, docs, seen, duplicates, truncated = [], [], set(), 0, 0
    for offset in offsets:
        query = urllib.parse.urlencode(dict(dataset='HuggingFaceFW/fineweb', config='sample-10BT', split='train', offset=offset, length=100))
        url = 'https://datasets-server.huggingface.co/rows?'+query
        path = raw_dir/f'rows_{offset}.json'
        if not path.exists():
            for attempt in range(3):
                try:
                    body = urllib.request.urlopen(url, timeout=30).read()
                    break
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2*(attempt+1))
            path.write_bytes(body)
        body = path.read_bytes()
        page = json.loads(body)
        pages.append(dict(url=url, file=str(path.relative_to(root)), sha256=sha256(body)))
        for item in page['rows']:
            if item.get('truncated_cells'):
                truncated += 1
                continue
            row = item['row']
            # Whitespace-normalized exact duplicates form one split group.
            group = sha256(' '.join(row['text'].split()).encode('utf-8'))
            if group in seen:
                duplicates += 1
                continue
            seen.add(group)
            bucket = int(group[:16], 16) % 10
            split = 'train' if bucket < 8 else ('validation' if bucket == 8 else 'test')
            docs.append(dict(id=row['id'], text=row['text'], split=split, group=group, row_index=item['row_idx'], source_url=row.get('url')))
        print(f'Fetched offset={offset}, complete documents so far={len(docs)}', flush=True)
    docpath = root/'documents.jsonl'
    docpath.write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in docs))
    files, counts = {}, {}
    for split in ['train','validation','test']:
        rows, ids = [], []
        for doc in docs:
            if doc['split'] != split:
                continue
            b = np.frombuffer(doc['text'].encode('utf-8')[:8192], dtype=np.uint8)
            for start in range(0, len(b)-length, length):
                rows.append(b[start:start+length+1])
                ids.append(doc['id'])
        if not rows:
            raise ValueError(f'Empty {split}')
        path = root/f'{split}.npz'
        np.savez_compressed(path, windows=np.stack(rows), doc_ids=np.array(ids))
        files[split] = dict(file=path.name, sha256=sha256(path.read_bytes()))
        counts[split] = dict(documents=sum(d['split']==split for d in docs), windows=len(rows), target_bytes=len(rows)*length)
    # Evaluation selects a deterministic round-robin across documents, never train.
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), dataset='HuggingFaceFW/fineweb', config='sample-10BT', source_pages=pages,
        source_revision='viewer snapshot; upstream revision not guaranteed', documents_sha256=sha256(docpath.read_bytes()),
        tokenization='raw UTF-8 bytes 0..255; no learned tokenizer, EOS or cross-document windows',
        max_bytes_per_document=8192, context=length, stride=length, split='sha256(normalized full text) mod 10: 0..7 train, 8 validation, 9 test',
        dedup='whitespace-normalized exact text only; near-duplicate leakage remains possible', duplicates_removed=duplicates,
        truncated_documents_removed=truncated, counts=counts, files=files)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(counts, indent=2), flush=True)


def load_windows(root, split, count=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    record = manifest['files'][split]
    path = root/record['file']
    if sha256(path.read_bytes()) != record['sha256']:
        raise ValueError(f'Snapshot hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        windows, ids = data['windows'].copy(), data['doc_ids'].copy()
    if count is not None:
        # Interleave documents in fixed hash order instead of evaluating one long page.
        groups = {doc: np.flatnonzero(ids==doc).tolist() for doc in sorted(set(ids), key=lambda x:sha256(x.encode()))}
        order = []
        depth = 0
        while len(order) < min(count, len(windows)):
            for indices in groups.values():
                if depth < len(indices):
                    order.append(indices[depth])
            depth += 1
        order = order[:count]
        windows, ids = windows[order], ids[order]
    return windows, ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/fineweb_pilot_v1')
    parser.add_argument('--context', type=int, default=64)
    args = parser.parse_args()
    prepare(args.output, args.context)
