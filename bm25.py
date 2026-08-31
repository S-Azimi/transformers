
from rank_bm25 import BM25Okapi  # pip install rank_bm25


docs = [
    "تراکنش شما به دلیل محدودیت موجودی انجام نشد",
    "رمز عبور شما تغییر کرد",
]
tokenized = [d.split() for d in docs]
bm25 = BM25Okapi(tokenized)

query = "تراکنش شما به دلیل محدودیت موجودی انجام نشد".split()
scores = bm25.get_scores(query)
print(f'the score is {scores}')


from rapidfuzz import fuzz # pip install rapidfuzz


s1 = "تراکنش شما به دلیل محدودیت موجودی انجام نشد"
s2 = "تراکنش من ناموفق شد"

# مقایسه اشتراک کلمات با در نظر گرفتن جابه‌جایی کلمات (Token Set Ratio)
score = fuzz.token_set_ratio(s1, s2)
print(score)  # عددی بین 0 تا 100