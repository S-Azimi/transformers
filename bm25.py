
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


s1 = "متوجه شدم که نیاز به صحبت با کارشناس دارید."
s2 = "شماره تماس میخوام زنگ بزنم پشتیبانی چون ربات جواب چیزی که میخوام رو نمیده بهم"

# مقایسه اشتراک کلمات با در نظر گرفتن جابه‌جایی کلمات (Token Set Ratio)
score_rapid = fuzz.token_set_ratio(s1, s2)
print(score_rapid)