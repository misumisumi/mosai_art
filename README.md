# mosaic_art_generator

ディレクトリにある写真郡からモザイクアートを作成する

```
❯ generate_mosaic_art -h                                                                                                                                                                                                                v3.12.8 (venv)
usage: generate_mosaic_art [-h] [-t TARGET] [-s SRC_DIR] [-o OUTPUT] [-j JOBS] [--src_find_pattern SRC_FIND_PATTERN] [--used_count USED_COUNT] [--zoom ZOOM] [--resize RESIZE [RESIZE ...]] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        ピクセルアートのリファレンス画像
  -s SRC_DIR, --src_dir SRC_DIR
                        ソースディレクトリ
  -o OUTPUT, --output OUTPUT
                        出力ファイル名
  -j JOBS, --jobs JOBS  使用するCPU数
  --src_find_pattern SRC_FIND_PATTERN
                        ソース画像を検索する時のパターン
  --used_count USED_COUNT
                        1つの画像を何回まで使えるか
  --zoom ZOOM           目標画像の倍率
  --resize RESIZE [RESIZE ...]
                        ソース画像のリサイズサイズ
  --device DEVICE       cpu or cuda
```
