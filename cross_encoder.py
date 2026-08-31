from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")

# scores = model.predict([
#     ("چرا تراکنش من ناموفق شده است؟", "به دلیل محدودیت موجودی، تراکنش شما انجام نشده است."),
#     ("چرا تراکنش من ناموفق شده است؟", "رمز عبور حساب کاربری شما با موفقیت تغییر یافت.")
# ])

scores = model.predict([
    ("چرا قد من کوتاه است؟", "بخاطر مسائل ژنتیکی"),
    ("چرا قد من کوتاه است؟", "بخاطر نخوردن شیر")
])

print("Scores:", scores)
