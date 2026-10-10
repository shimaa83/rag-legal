
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "ما هو سن الرشد في القانون المدني؟",
    "متى تبدأ الشخصية القانونية للإنسان؟",
    "ما هي أركان العقد؟",
    "هل يجوز التعاقد مع القاصر؟",
    "ما هو أثر الغلط على صحة العقد؟",
    "متى يكون العقد باطلًا؟",
]


class LegalRAGUser(HttpUser):
    # Each request runs retrieval + LLM generation.
    wait_time = between(1, 3)

    @task
    def ask(self):
        with self.client.post(
            "/ask",
            json={"question": random.choice(QUESTIONS)},
            timeout=300,
            name="/ask",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return

            try:
                data = response.json()
            except ValueError:
                response.failure("Response is not valid JSON")
                return

            if not data.get("answer", "").strip():
                response.failure("Empty answer")
            else:
                response.success()