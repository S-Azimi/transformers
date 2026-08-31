from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.100:8000/v1",
    api_key="empty",  # vLLM does not require a real key by default
)

response = client.chat.completions.create(
    model="gemma-4-12B-it-AWQ-INT4",
    messages=[
        {
            "role": "user",
            "content": "Explain PCA in three sentences."
        }
    ],
    temperature=0.7,
    max_tokens=300,
)

print(response.choices[0].message.content)