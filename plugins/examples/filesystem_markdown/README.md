# filesystem_markdown Example Plugin

This example plugin scans a folder for `*.md` files and emits CoLibri document envelopes as JSONL.
It also supports incremental sync by reading `cursor.last_scan_at` from stdin and emitting a final cursor control line.

## Run

```bash
cd plugins/examples/filesystem_markdown
python3 plugin.py <<'JSON'
{
  "config": {
    "root_path": "/path/to/markdown",
    "classification": "internal"
  }
}
JSON
```

The output is one JSON envelope per markdown file.
