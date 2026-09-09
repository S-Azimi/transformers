import csv
import json
from openai import OpenAI

### set the LLM model ####################################################################################
client = OpenAI(
    base_url="http://192.168.0.10:8000/v1",
    api_key="empty",  # vLLM does not require a real key by default
)

# --- 1. File Path Configuration ---
INPUT_CSV_PATH = "data/temp_input.csv"
OUTPUT_CSV_PATH = "data/evaluations4.csv"


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

* Subject / Context: {subject}
* User Ticket: {q}
* Agent Response: {a}

---

You MUST evaluate the response in TWO strictly separated stages.

STAGE 1 — extract main_answer from Agent Response

Split the Agent Response into semantic sentences or clauses.

For each sentence/clause, classify it as either:

A. PARAPHRASE / ACKNOWLEDGEMENT

A sentence or clause is PARAPHRASE of question if:

repeats information already stated by the user,
summarizes the user's problem,
confirms understanding,
expresses empathy or acknowledgement,
reformulates the ticket using different words.

Examples:

"متوجه شدم که شهر تولد شما در لیست وجود ندارد."
"همانطور که فرمودید، در مرحله آخر با خطا مواجه می‌شوید."
"درک می‌کنم که امکان ورود به دیما را ندارید."
"با توجه به توضیحات شما..."
"متوجه شدم که مشکل شما عدم موفقیت در احراز هویت است."

These parts contain NO answer and NO problem resolution.

B. main_answer

an answer,
a solution,
an instruction,
troubleshooting relevant to the reported problem,
necessary clarification,
genuinely useful new information.

DELETE PARAPHRASE / ACKNOWLEDGEMENT and just keep main_answer

JUST use main_answer for next step

## STEP 2 — EVALUATE THE main_answer

### 1. Completeness

Score from 1 to 10.

#### Goal

Identify all distinct user intents, questions, problems, or requested actions in the main_answer and determine whether the main_answer of the Agent Response addresses each one.

#### Scoring Rubric

* **10:** Fully complete. Addresses all important user intents with clear and actionable information.
* **1:** Does not substantively address the user's main request.

#### Important

Merely repeating the user's problem does NOT count as addressing it.

Example:

Ticket:
"من درخواست وام شایان دادم ولی انجام نشده و حتی نمی تونم وارد دیما بشم"

Response:
"در صورت ثبت درخواست وام شایان و عدم دریافت پاسخ تا کنون، گاهی لازم است تا چند روز صبر کنید چون فرایند استعلام ها ممکن است طولانی شود."

Result:
The loan-delay issue is addressed, but the Dima login issue is completely missed. Therefore, the response is incomplete.

Example:

Ticket:
"شماره تماس میخوام زنگ بزنم پشتیبانی چون ربات جواب چیزی که میخوام رو نمیده بهم"

Response:
"متوجه شدم که نیاز به صحبت با کارشناس دارید."

The requested support phone number was not provided.

---

### 2. Relevancy

Score from 1 to 10.

#### Goal

Evaluate whether the main_answer of the response is directly related to the user's actual issue, product, service, entity, and request.

#### Scoring Rubric

* **10:** Directly and specifically relevant to the exact issue and product/service.
* **1:** Irrelevant, discusses the wrong product/service, nonsensical, or contains no substantive response after paraphrase removal.

#### Important

Do NOT assign a high relevancy score merely because the response paraphrases the user's ticket correctly.

Example:

Ticket:
"من می خوام وام میکرولون بگیرم اما بعد از ثبت درخواست هنوز واریز نشده"

Response:
"واریز وام شایان ممکن است تا بعد از دریافت نتایج استعلامات لازم، طولانی شود"

Result:
Low relevancy because the response discusses **Shayan Loan** instead of **Microloan**.

Example:

Ticket:
"من شهر تولدم را در فهرست آدرس ها پیدا نمی‌ کنم"
Response:
"تاریخ تولد خود را مطابق آنچه که در کارت ملی شما وجود دارد، وارد کنید."

Result:
Low relevancy because the response discusses **‌date of birth** instead of **place of birth**.


### STATUS CLASSIFICATION

Choose exactly ONE of the following categories:

#### "more_info_required"

Use when:

* The user's core issue is understandable.
* The agent correctly identifies a specific piece of additional information that is genuinely needed to investigate or answer the issue.
* The agent explicitly asks the user to provide that information.

Examples of useful missing information:

* Exact error message
* Account or transaction identifier
* Date/time of transaction
* Which account/card/service is affected
* Screenshot or other diagnostic information

Example:

Ticket:
"کارت به کارت انجام نمیشه"

Response:
"لطفاً مشخص کنید کارت به کارت از کدام حساب شما انجام نمی‌شود و دقیقاً چه پیامی دریافت می‌کنید."

Status:
"more_info_required"

Another example:

Ticket:
"تمامی استعلام ها تایید شده جز مرحله آخر که با خطا مواجه میشه"

Response:
"متوجه شدم که تمامی استعلام‌ها تأیید شده‌اند و فقط مرحله آخر خطا دارد. لطفاً متن دقیق خطایی که در مرحله آخر مشاهده می‌کنید ارسال کنید."

### "ticket_not_clear"

Use when:

* The **ticket itself** is too vague, incomplete, ambiguous, or unintelligible to identify what the user's actual problem or request is.
* The agent appropriately asks the user to explain what they mean.

Example:

Ticket:
"اصلاً نمیشه"

Response:
"لطفاً مشخص کنید چه کاری را انجام می‌دهید و در کدام مرحله با مشکل مواجه می‌شوید."

Status:
"ticket_not_clear"

Important distinction:

If the user's problem is understandable but a technical detail is needed for troubleshooting, use `"more_info_required"`, NOT `"ticket_not_clear"`.

---

### "not_found"

Use when the agent fails to provide a substantive answer.

This includes cases where:

1. After removing paraphrase, acknowledgment the main_answer is not meaningful.

Example:

Ticket:
"وام من هنوز واریز نشده"

Response:
"متوجه شدم که وام شما هنوز واریز نشده است."

in main_answer the paraphrase is removed so nothing is remaing.

Status:
"not_found"

2. The agent restates the user's issue without answering or taking a meaningful next step.

3. The agent explicitly indicates that it cannot provide the answer or lacks the required knowledge/data.

4. The agent simply redirects the user to another support channel instead of answering the question.

Example:

Ticket:
"میخوام امتیاز حساب شایانم رو به مادرم منتقل کنم، چیکار کنم؟"

Response:
"برای دریافت پاسخ با پشتیبانی تماس بگیرید."

Status:
"not_found"

5. The response consists only of generic statements that do not resolve the issue or request useful diagnostic information.

Important:
A **specific and necessary request for additional information** is `"more_info_required"`, not `"not_found"`.

---

### "valid"

Use when:

* The main_answer rationally and directly answers the user's ticket.
* main_answer provides useful information, instructions, explanation, or a resolution.
* No essential additional information from the user is required before giving the answer.


The response does NOT need to be perfect to be `"valid"`. Minor completeness or relevancy issues should be reflected in the numeric scores.

---

## STATUS DECISION PRIORITY

Use the following logic:

1. First create main_answer
2. If main_answer is less than 2 words → `"not_found"`.
3. If the ticket itself is too unclear to determine the user's issue and the agent asks for clarification → `"ticket_not_clear"`.
4. If the issue is understandable, but specific additional information is genuinely required and the agent asks for it → `"more_info_required"`.
5. If the agent provides a substantive answer/resolution → `"valid"`.
6. If the agent fails to answer and instead only redirects, admits lack of information, or gives no useful next step → `"not_found"`.

---

## OUTPUT FORMAT

Return strictly one valid JSON object.

Do not include explanations, markdown, comments, or additional text.

Use exactly this schema:

{{
"completeness_score": <integer from 1 to 10>,
"relevancy_score": <integer from 1 to 10>,
"status": "<more_info_required | ticket_not_clear | not_found | valid>"
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