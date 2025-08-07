<div align="center" style="text-align: center;">
  <h1>Facelapse</h1>
  <img src="https://raw.githubusercontent.com/walkersutton/facelapse/refs/heads/main/facelapse.gif"/ style="width: 269px;">
</div>

## Requirements
* Python
* [uv](https://docs.astral.sh/uv/getting-started/installation/)

## How To
1. add images of your face to the `raws` directory
3. execute `uv run main.py`
4. open `facelapse.gif`

## CLI args
* `--sort`
  [date (default), happiness, filename]
* `--anchor`
  [face (default), left-eye]
* `--draw-date`
  __no value__

example w/ args: 
`uv run main.py --sort happiness --anchor left-eye --draw-date`

  

