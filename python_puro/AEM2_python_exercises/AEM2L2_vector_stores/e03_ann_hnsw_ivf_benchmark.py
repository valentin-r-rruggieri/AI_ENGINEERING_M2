"""E03: comparar Flat, HNSW e IVF sobre los mismos vectores."""
from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from common import deterministic_embedding, normalize, split_words, write_json

root=Path(__file__).parent
chunks=split_words((root/"data"/"corpus.txt").read_text(encoding="utf-8")*20,25,5,"corpus")
matrix=np.asarray([normalize(deterministic_embedding(c.content)) for c in chunks],dtype="float32")
query=np.asarray([normalize(deterministic_embedding("auditoría y autenticación"))],dtype="float32")
indices={"flat":faiss.IndexFlatIP(matrix.shape[1]),"hnsw":faiss.IndexHNSWFlat(matrix.shape[1],16,faiss.METRIC_INNER_PRODUCT)}
ivf=faiss.IndexIVFFlat(faiss.IndexFlatIP(matrix.shape[1]),matrix.shape[1],min(8,len(chunks)),faiss.METRIC_INNER_PRODUCT); ivf.train(matrix); indices["ivf"]=ivf
report={}
for name,index in indices.items():
    index.add(matrix); started=time.perf_counter(); scores,ids=index.search(query,5)
    report[name]={"latency_ms":round((time.perf_counter()-started)*1000,4),"ids":ids[0].tolist(),"scores":[float(x) for x in scores[0]]}
write_json(root/"data"/"benchmark.json",report); print(report)

