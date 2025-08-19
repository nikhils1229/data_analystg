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
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},  # forces JSON
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data analyst agent.\n"
                        "Your ONLY output must be a valid JSON object.\n"
                        "Do not include explanations, comments, or text outside JSON.\n"
                        "Infer the keys/values based on the user's question.\n"
                    )
                },
                {"role": "user", "content": question}
            ]
        )

        # This is guaranteed to be JSON, but still wrap in try
        content = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(content)
        except Exception:
            # fallback: wrap raw content
            parsed = {"raw": content}

        return parsed

    except Exception as e:
        return {"error": str(e)}
