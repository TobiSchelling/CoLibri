# Backup and Restore (Portable `COLIBRI_HOME`)

CoLibri’s primary state lives under `COLIBRI_HOME` (default: `~/.local/share/colibri`).

This directory is designed to be portable:
- Copying it to another machine should preserve `manifest.json` (active generation pointer), `metadata.db`, and the canonical corpus.
- Index directories may be large; include them if you want a “no re-index” restore.

## Backup

Set an explicit root to avoid ambiguity:

```bash
export COLIBRI_HOME="$HOME/.local/share/colibri"
```

Create a tarball:

```bash
tar -C "$(dirname "$COLIBRI_HOME")" -czf "colibri-home-backup.tgz" "$(basename "$COLIBRI_HOME")"
```

## Restore

Extract to the target location:

```bash
mkdir -p "$(dirname "$COLIBRI_HOME")"
tar -C "$(dirname "$COLIBRI_HOME")" -xzf "colibri-home-backup.tgz"
```

Verify:

```bash
colibri doctor
colibri serve --check
```

## Notes

- `colibri migrate` creates pre-migration copies of `manifest.json` and `metadata.db` under `$COLIBRI_HOME/backups/`.
- If you restore without `indexes/`, you can rebuild on the target machine with:
  - `colibri index --canonical --force`

