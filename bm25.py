
from rank_bm25 import BM25Okapi

docs = [
    "تراکنش شما به دلیل محدودیت موجودی انجام نشد",
    "رمز عبور شما تغییر کرد",
]
tokenized = [d.split() for d in docs]
bm25 = BM25Okapi(tokenized)

query = "تراکنش من ناموفق شد".split()
scores = bm25.get_scores(query)