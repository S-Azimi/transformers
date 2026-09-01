import csv
import json
from openai import OpenAI

### set the LLM model ####################################################################################
client = OpenAI(
    base_url="http://192.168.0.10:8000/v1",
    api_key="empty",  # vLLM does not require a real key by default
)

# --- 1. File Path Configuration ---
INPUT_CSV_PATH = "data/tickets -labeling-200-14050607.csv"
OUTPUT_CSV_PATH = "data/evaluations3.csv"


# --- 2. Prompts Setup ---
SYSTEM_PROMPT = (
    "You are an expert AI Quality Assurance Judge specialized in customer support ticketing systems. "
    "Your task is to evaluate the quality of an AI Agent's response to a user ticket based on three strict criteria: "
    "Completeness, Relevancy, and Status Classification. Always output strictly valid JSON."
)

# --- 3. CSV Reading & Writing Setup ---
with open(INPUT_CSV_PATH, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter=';')
    
    # Define updated headers
    fieldnames = [
        'ticket_id', 
        'subject', 
        'question', 
        'answer', 
        'completeness_score', 
        'relevancy_score', 
        'status', 
    ]
    
    with open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        # --- 4. Process Each Ticket ---
        for row in reader:
            ticket_id = row['ticket_id']
            subject = row.get('subject', '')
            q = row['question']
            a = row['answer']

            print(f"Evaluating Ticket ID: {ticket_id}...")

            user_prompt = f"""Evaluate the following customer support Ticket and Agent Response.

### INPUT DATA
- Subject / Context: {subject}
- User Ticket: {q}
- Agent Response: {a}

### EVALUATION CRITERIA & GUIDELINES

1. Completeness (Score: 1 - 10):
- Goal: Extract all distinct user intents/questions and verify if the response addresses each one.
- Scoring Rubric:
  * 9-10: Fully complete. Addresses 100% of user intents with actionable, clear details.
  * 6-8: Partially complete. Misses a minor intent or provides an incomplete resolution for one aspect.
  * 3-5: Major omissions. Addresses only one intent while ignoring other critical requests.
  * 1-2: Completely fails to address core requirements or misses all primary intents.
- Examples:
  * Ticket: "من درخواست وام شایان دادم ولی انجام نشده و حتی نمی تونم وارد دیما بشم"
    Response: "در صورت ثبت درخواست وام شایان و عدم دریافت پاسخ تا کنون، گاهی لازم است تا چند روز صبر کنید چون فرایند استعلام ها ممکن از طولانی شود."
    Completeness: Incomplete because the login issue into Dima was completely missed.
  * Ticket: "شماره تماس میخوام زنگ بزنم پشتیبانی چون ربات جواب چیزی که میخوام رو نمیده بهم"
    Response: "متوجه شدم که نیاز به صحبت با کارشناس دارید."
    Completeness: Incomplete because the support phone number was not provided.

2. Relevancy (Score: 1 - 10):
- Goal: Check if the response directly addresses the exact topic, entities, and services mentioned without confusing products.
- Scoring Rubric:
  * 9-10: Highly relevant to the specific context and product.
  * 6-8: Mostly relevant with minor generic filler text.
  * 3-5: Poor relevance, talks about wrong product or gives off-target advice.
  * 1-2: Completely irrelevant or nonsensical.
- Example:
  * Ticket: "من می خوام وام میکرولون بگیرم اما بعد از ثبت درخواست هنوز واریز نشده"
    Response: "واریز وام شایان ممکن است تا بعد از دریافت نتایج استعلامات لازم، طولانی شود"
    Relevancy: Irrelevant product (Microloan vs Shayan Loan).

3. Status Classification:
Choose exactly ONE category:
- "more_info_required": The agent correctly identifies that extra parameters (account number, error message, etc.) are needed from the user.
  (Example: Ticket: "کارت به کارت انجام نمیشه" -> Response: "لطفاً مشخص کنید که کارت به کارت از کدام حساب شما انجام نمی‌شود و دقیقاً چه پیامی دریافت می‌کنید.")
- "ticket_not_clear": The ticket is too short, vague, or unintelligible, and the agent asks for clarification.
  (Example: Ticket: "اصلاً نمیشه" -> Response: "لطفاً مشخص کنید که چه چیزی انجام نمی‌شود")
- "not_found": The agent lacks information to answer (e.g., asks to call support or mentions lack of data).
  (Example: Ticket: "میخوام امتیاز حساب شایانم رو به مادرم منتقل کنم، چیکار کنم؟" -> Response: "برای دریافت پاسخ با پشتیبانی تماس بگیرید.")
- "valid": The response is rational, substantive, and directly answers the ticket.

### OUTPUT FORMAT
Output strictly as a valid JSON object matching this schema:
{{
  "completeness_score": <1-10>,
  "relevancy_score": <1-10>,
  "status": "<more_info_required | ticket_not_clear | not_found | valid>",
}}
"""

            try:
                # --- LLM API Call ---
                response = client.chat.completions.create(
                    model="gemma-4-12B-it-AWQ-INT4",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=450,
                )

                # Parse JSON evaluation output
                raw_content = response.choices[0].message.content
                eval_data = json.loads(raw_content)

                # Combine original row data with evaluation output
                output_row = {
                    'ticket_id': ticket_id,
                    'subject': subject,
                    'question': q,
                    'answer': a,
                    'completeness_score': eval_data.get('completeness_score', ''),
                    'relevancy_score': eval_data.get('relevancy_score', ''),
                    'status': eval_data.get('status', ''),
        
                }

            except Exception as e:
                print(f"Error processing ticket {ticket_id}: {e}")
                output_row = {
                    'ticket_id': ticket_id,
                    'subject': subject,
                    'question': q,
                    'answer': a,
                    'completeness_score': None,
                    'relevancy_score': None,
                    'status': 'error',
            
                }

            # Write the result row immediately to file
            writer.writerow(output_row)

print(f"\nEvaluation complete! Results saved to: {OUTPUT_CSV_PATH}")













# prompt = f"""
# You are an evaluation assistant. Evaluate the following Question and Answer based on:
# 1. Completeness
# 2. Quality
# 3. Relevancy
# 4. Validation

# Question: {q}
# Answer: {a}

# Provide your evaluation strictly as a valid JSON object matching this schema:
# {{
#     "completeness_score": <1-10>,
#     "quality_score": <1-10>,
#     "relevancy_score": <1-10>,
#     "validation_score": <1-10>,
#     "total_score": <1-10>,
#     "feedback": "<brief explanation>"
# }}
# Do not include any text outside the JSON object.
# """

# response = client.chat.completions.create(
#     model="gemma-4-12B-it-AWQ-INT4",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
#         {"role": "user", "content": prompt}
#     ],
#     response_format={"type": "json_object"},
#     temperature=0.2,  # Lower temperature is recommended for structured JSON extraction
#     max_tokens=300,
# )

# # Extract and parse the JSON response
# raw_content = response.choices[0].message.content
# result_json = json.loads(raw_content)

# print(result_json)
# print(f"Total Score: {result_json.get('total_score')}")