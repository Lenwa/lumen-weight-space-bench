# Weight-space tool entry points

See the [repository README](../../README.md) for setup, commands, experimental limits, and provenance. The original directory layout is retained so unittest discovery and sibling imports continue to work:

```sh
python -B -m unittest discover -s tools/weight-space -p test_bench.py -v
```

The Python files and vendor notice are preserved byte-for-byte from the source snapshot. The source README was replaced to remove private project context and to document standalone use.
