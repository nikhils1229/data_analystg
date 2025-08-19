# app/llm.py
import json
from openai import OpenAI

client = OpenAI()

def call_llm(question: str) -> dict:
    """
    Directly send the question to OpenAI and let the LLM decide
    the schema + values. Always enforce JSON response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # small + cheap; can swap
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst agent.\n"
                        "Always respond with a **single JSON object only**, no text outside JSON.\n"
                        "The keys and structure must be inferred from the user question.\n"
                        "Example:\n"
                        "Q: 'Analyze sample-weather.csv...'\n"
                        "A: {\n"
                        '  "average_temp_c": 5.1,\n'
                        '  "max_precip_date": "2024-01-06",\n'
                        '  "min_temp_c": 2,\n'
                        '  "temp_precip_correlation": 0.041,\n'
                        '  "average_precip_mm": 0.9,\n'
                        '  "temp_line_chart": "data:image/png;base64,...",\n'
                        '  "precip_histogram": "data:image/png;base64,..."\n'
                        "}\n\n"
                        "If the question is unrelated to datasets/images, still return JSON with relevant keys."
                    )
                },
                {"role": "user", "content": question}
            ]
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        return {"error": str(e)}
