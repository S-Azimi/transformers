from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.10:8000/v1",
    api_key="empty",  # vLLM does not require a real key by default
)
q="شماره تماس میخوام زنگ بزنم پشتیبانی چون ربات جواب چیزی که میخوام رو نمیده بهم"
a="متوجه شدم که نیاز به صحبت با کارشناس دارید."

q="احراز هویت انجام دادم تایید هم شد ولی الان باز میگه احراز کن مراحلشم باز نمیشه دیگه"
a="متوجه شدم؛ شما فرآیند احراز هویت را تکمیل کرده‌اید و نتیجه آن نیز تأیید شده است، اما اکنون هنگام استفاده از خدمات، پیام نیاز به احراز هویت دریافت می‌کنید و امکان دسترسی به مراحل آن وجود ندارد. این موضوع ممکن است ناشی از تأخیر در به‌روزرسانی وضعیت در سامانه یا یک اختلال موقت باشد. لطفاً متن دقیق پیامی که نمایش داده می‌شود را اعلام نمایید تا بتوانیم علت را بررسی کنیم."


import json

prompt = f"""
You are an evaluation assistant. Evaluate the following Question and Answer based on:
1. Completeness
2. Quality
3. Relevancy
4. Validation

Question: {q}
Answer: {a}

Provide your evaluation strictly as a valid JSON object matching this schema:
{{
    "completeness_score": <1-10>,
    "quality_score": <1-10>,
    "relevancy_score": <1-10>,
    "validation_score": <1-10>,
    "total_score": <1-10>,
    "feedback": "<brief explanation>"
}}
Do not include any text outside the JSON object.
"""

response = client.chat.completions.create(
    model="gemma-4-12B-it-AWQ-INT4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
        {"role": "user", "content": prompt}
    ],
    response_format={"type": "json_object"},
    temperature=0.2,  # Lower temperature is recommended for structured JSON extraction
    max_tokens=300,
)

# Extract and parse the JSON response
raw_content = response.choices[0].message.content
result_json = json.loads(raw_content)

print(result_json)
print(f"Total Score: {result_json.get('total_score')}")