from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.10:8000/v1",
    api_key="empty",  # vLLM does not require a real key by default
)
q="شماره تماس میخوام زنگ بزنم پشتیبانی چون ربات جواب چیزی که میخوام رو نمیده بهم"
a="متوجه شدم که نیاز به صحبت با کارشناس دارید."

q="احراز هویت انجام دادم تایید هم شد ولی الان باز میگه احراز کن مراحلشم باز نمیشه دیگه"
a="متوجه شدم؛ شما فرآیند احراز هویت را تکمیل کرده‌اید و نتیجه آن نیز تأیید شده است، اما اکنون هنگام استفاده از خدمات، پیام نیاز به احراز هویت دریافت می‌کنید و امکان دسترسی به مراحل آن وجود ندارد. این موضوع ممکن است ناشی از تأخیر در به‌روزرسانی وضعیت در سامانه یا یک اختلال موقت باشد. لطفاً متن دقیق پیامی که نمایش داده می‌شود را اعلام نمایید تا بتوانیم علت را بررسی کنیم."

response = client.chat.completions.create(
    model="gemma-4-12B-it-AWQ-INT4",
    messages=[
        {
            "role": "user",
            "content": "check the completeness, quality and relevancy and validation of the question which is " + q + "and answer is " + a + "and give me the total result as a score between 1 to 10 where 1 is the lowest score. "
        }
    ],
    temperature=0.7,
    max_tokens=300,
)

print(response.choices[0].message.content)