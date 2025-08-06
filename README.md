# Facelapse

## Requirements
* Python
* [uv](https://docs.astral.sh/uv/getting-started/installation/)

## How To
1. create a directory, `raws`, and add images of your face (images will be compiled in numerical order)
2. execute `uv run main.py`
3. open `facelapse.gif`

## Notes
* `main.py` is currently configured to anchor the timelapse on the left eye - you'll have to make some modifications to anchor facelapse on a different facial feature

## Appendix

### Convert HEIC files in current directory -> jpg
```
for f in *.HEIC; do sips -s format jpeg "$f" --out "${f%.*}.jpg"; done;
```
