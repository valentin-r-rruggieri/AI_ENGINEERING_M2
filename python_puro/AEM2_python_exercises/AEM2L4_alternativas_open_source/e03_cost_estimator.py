"""E03: estimar costo de embeddings sin fijar precios en código."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write_json

parser=argparse.ArgumentParser()
parser.add_argument("--price-per-million",type=float,default=0.02)
args=parser.parse_args()
tokens=250_000
estimate={"input_tokens":tokens,"estimated_usd":round(tokens/1_000_000*args.price_per_million,4)}
write_json(Path(__file__).parent/"data"/"cost_estimate.json",estimate); print(estimate)

