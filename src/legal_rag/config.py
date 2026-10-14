import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_JSON_PATH = RAW_DATA_DIR / "civil_code.json"
PROCESSED_JSON_PATH = PROCESSED_DATA_DIR / "civil_code.json"

# DVC tracking
CIVIL_CODE_PDF_PATH = RAW_DATA_DIR / "egyptian_civil_code.pdf"
CIVIL_CODE_DVC_PATH = RAW_DATA_DIR / "egyptian_civil_code.pdf.dvc"

# ============================================================
# Arabic Text / Legal Code
# ============================================================

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"

# Articles 54-80 were repealed.
REPEALED_ARTICLES = set(range(54, 81))

# ============================================================
# MLflow / DagsHub
# ============================================================

USE_DAGSHUB = os.getenv("USE_LOCAL_MLFLOW") != "1"
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_USER", "shimaa83")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO", "rag-legal")

LOCAL_MLFLOW_URI = "http://localhost:5000"
DAGSHUB_MLFLOW_URI = (
    f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
)
MLFLOW_TRACKING_URI = DAGSHUB_MLFLOW_URI if USE_DAGSHUB else LOCAL_MLFLOW_URI

MLFLOW_EXPERIMENT_NAME = "legal-rag_using_ranker_50_questions"
MLFLOW_RUN_NAME = None

# ============================================================
# Experiment Settings
# ============================================================

NUM_QUESTIONS = 50

# Retrieval (can be overridden with environment variables)
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "20"))

# Redis cache (fail-open: if Redis is down the pipeline runs without cache)
# Disable without editing code:  RAG_CACHE=0
CACHE_ENABLED = os.getenv("RAG_CACHE", "1").lower() in {"1", "true", "yes"}
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "1.0"))
CACHE_PREFIX = os.getenv("RAG_CACHE_PREFIX", "legal_rag")
# Bump this when prompts or the reranker change to invalidate old entries.
# (A change in the processed data file invalidates the cache automatically.)
CACHE_VERSION = os.getenv("RAG_CACHE_VERSION", "v3")
CACHE_TTL_RETRIEVAL = int(os.getenv("RAG_CACHE_TTL_RETRIEVAL", str(7 * 24 * 3600)))
CACHE_TTL_ANSWER = int(os.getenv("RAG_CACHE_TTL_ANSWER", str(24 * 3600)))

# Load the reranker / search index / Ollama models once when the API starts,
# so the first user request is as fast as the following ones.
# Disable without editing code:  RAG_WARMUP=0
WARMUP_ON_STARTUP = os.getenv("RAG_WARMUP", "1").lower() in {"1", "true", "yes"}

# Generator
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "qwen2.5:3b")

# Verification (local model via Ollama, no external API)
# True  -> the draft answer is reviewed by VERIFIER_MODEL (slower, stricter)
# False -> the draft answer is returned as is (faster)
# Override without editing code:  RAG_VERIFY=0
ENABLE_VERIFICATION = os.getenv("RAG_VERIFY", "1").lower() in {"1", "true", "yes"}
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "qwen2.5:3b")

# Judge (used by the evaluation experiments only, not by the RAG pipeline)
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gemini").lower()
API_JUDGE_MODEL = os.getenv("API_JUDGE_MODEL", "gemini-3.5-flash-lite")
JUDGE_BASE_MODEL = os.getenv("JUDGE_BASE_MODEL", "qwen2.5:3b")
JUDGE_NUM_CTX = int(os.getenv("JUDGE_NUM_CTX", "8192"))

# ============================================================
# Prompts
# ============================================================

INSUFFICIENT_CONTEXT = "لا توجد معلومات كافية في المواد المسترجعة."

# 1. Initial answer generation


PROMPT_TEMPLATE = """
أنت مساعد قانوني متخصص في القانون المدني المصري.
أجب اعتمادًا حصريًا على السياق القانوني المسترجع والمقدم لك.

قواعد اللغة الإلزامية:
1. حدد لغة السؤال أولًا، ثم أجب باللغة نفسها فقط.
2. إذا كان السؤال باللغة العربية، فاكتب الإجابة بالعربية فقط.
3. إذا كان السؤال باللغة الإنجليزية، فاكتب الإجابة بالإنجليزية فقط.
4. إذا كان السؤال مختلطًا، فاستخدم اللغة الغالبة فيه.
5. لا تترجم الإجابة إلى لغة أخرى.
6. لا تكرر الإجابة بلغتين تحت أي ظرف، حتى لو كان السياق
   القانوني يحتوي على نصوص عربية وإنجليزية.
7. لا تسمح للغة السياق القانوني أو لغة أي مصدر بتحديد
   لغة الإجابة؛ لغة السؤال هي الأساس.

قواعد الإجابة القانونية:

1. افهم السؤال وحدد المطلوب منه بدقة قبل صياغة الإجابة.

2. ابحث في السياق المسترجع عن المادة أو المواد القانونية
   التي تجيب مباشرة عن السؤال.

3. أجب عن السؤال المطروح نفسه، ولا تستبدل السؤال العام
   بسؤال جزئي أو بحالة قانونية خاصة دون توضيح ذلك.

4. إذا كان السؤال عامًا، فغطِّ الأحكام الأساسية التي يدعمها
   السياق المسترجع، ولا تكتفِ بذكر حالة واحدة إذا كان السياق
   يحتوي على أحكام أخرى ذات صلة.

5. إذا كان السؤال يتطلب حكمًا محددًا، مثل نعم أو لا،
   فابدأ بالحكم مباشرة، ثم اشرح السبب بإيجاز.

6. استند حصريًا إلى النصوص القانونية الموجودة في السياق.
   لا تضف أحكامًا أو شروطًا أو استثناءات من ذاكرتك أو من
   معلومات خارج السياق المسترجع.

7. لا تفترض أن المعلومات التي تعرفها مسبقًا عن القانون
   موجودة في السياق. النصوص المسترجعة هي مرجعك للإجابة.

8. لا تخلط بين المواد القانونية المختلفة، ولا تنسب حكمًا
   إلى مادة لا تتضمنه.

9. اذكر أرقام المواد كما وردت في السياق فقط.
   لا تخترع أرقام مواد أو أسماء قوانين أو اقتباسات قانونية.

10. حافظ على المعنى القانوني للنص عند إعادة صياغته.
    لا تغيّر الشروط أو الأرقام أو المدد أو الاستثناءات الواردة
    في النصوص المسترجعة.

11. إذا تضمن السياق نصًا قانونيًا ملغى، فلا تعرضه باعتباره
    حكمًا نافذًا. وضّح أنه ملغى إذا كانت هذه المعلومة مثبتة
    في السياق.

12. إذا كانت المعلومات المسترجعة لا تكفي للإجابة عن السؤال،
    فلا تخمّن ولا تكمل الإجابة اعتمادًا على الذاكرة.
    استخدم العبارة التالية بحسب لغة السؤال:

    بالعربية:
    "{insufficient_context}"

    بالإنجليزية:
    "The retrieved legal context does not contain enough
    information to answer this question."

13. إذا كان السياق يدعم جزءًا من السؤال فقط، فاذكر ما يمكن
    إثباته من النصوص، ووضّح أن بقية التفاصيل لا تتوافر في
    السياق المسترجع.

14. لا تدّعِ أن جميع جوانب السؤال قد عولجت إذا كانت المواد
    المسترجعة لا تغطيها بالكامل.

15. راجع كل رقم ومدة وشرط ورقم مادة قبل إخراج الإجابة،
    وتأكد من أنها مدعومة بالنصوص المسترجعة.

تنسيق الإجابة:
- ابدأ بالإجابة المباشرة عن السؤال.
- اشرح الحكم القانوني بإيجاز ووضوح.
- اذكر رقم المادة بجوار الحكم متى كان متاحًا في السياق.
- استخدم لغة السؤال فقط، دون ترجمة أو تكرار بلغة أخرى.
- لا تعرض خطوات التفكير الداخلية.
- لا تضف مقدمة عامة أو معلومات غير ضرورية.
- لا تذكر أن الإجابة مؤكدة قانونيًا إذا لم يدعم السياق ذلك.
- أخرج الإجابة النهائية فقط دون تعليق على طريقة عملك.

السياق القانوني المسترجع:
{context}

السؤال:
{question}

الإجابة:
""".strip()


# 2. Verification (runs on VERIFIER_MODEL, a small local model:
#    short explicit rules and output-only instructions work best)


VERIFICATION_PROMPT_TEMPLATE = """
أنت مراجع قانوني لنظام استرجاع وتوليد الإجابات (RAG)
المتخصص في القانون المدني المصري.

مهمتك مراجعة الإجابة المقترحة وإنتاج إجابة نهائية دقيقة،
اعتمادًا حصريًا على النصوص القانونية الموجودة في السياق المسترجع.

قواعد اللغة الإلزامية:
1. حدد لغة السؤال واستخدمها وحدها في الإجابة النهائية.
2. إذا كان السؤال بالعربية، فأخرج الإجابة بالعربية فقط.
3. إذا كان السؤال بالإنجليزية، فأخرج الإجابة بالإنجليزية فقط.
4. إذا كان السؤال مختلطًا، فاستخدم اللغة الغالبة فيه.
5. لا تترجم الإجابة إلى لغة أخرى أثناء المراجعة.
6. احذف أي ترجمة أو تكرار للإجابة بلغة أخرى غير لغة السؤال.
7. لا تحتفظ بنص إنجليزي لمجرد أنه موجود في الإجابة المقترحة
   أو في السياق القانوني.
8. لا تستخدم مصطلحات أجنبية إلا عند الضرورة، ولا تقدّم
   ترجمة كاملة للإجابة بلغة ثانية.

قواعد المراجعة القانونية:

1. أعد تقييم السؤال بالرجوع إلى النصوص القانونية المسترجعة.
   لا تفترض أن الإجابة المقترحة صحيحة.

2. اعتبر النصوص القانونية المسترجعة المرجع الأساسي،
   وليس الإجابة المقترحة أو معلوماتك السابقة.

3. تحقق من أن الإجابة تجيب فعلًا عن السؤال المطروح،
   ولا تكتفي بحالة قانونية جزئية إذا كان السؤال أعمّ منها.

4. إذا كان السؤال عامًا، فاذكر الأحكام الأساسية التي يدعمها
   السياق، ولا تعمم حكمًا خاصًا على جميع الحالات.

5. تحقق من ارتباط كل مادة قانونية بالسؤال وبالحكم المنسوب إليها.

6. لا تضف أحكامًا أو شروطًا أو استثناءات أو معلومات
   غير موجودة في السياق المسترجع.

7. لا تغيّر أي رقم أو عمر أو مدة أو شرط أو رقم مادة.
   طابق هذه البيانات مع النصوص المسترجعة.

8. لا تنسب حكمًا إلى مادة لا تتضمن هذا الحكم.

9. لا تعرض حكم مادة ملغاة على أنه حكم نافذ.
   إذا كان إلغاء المادة مثبتًا في السياق، فلا تقدّمها
   باعتبارها نصًا ساريًا.

10. صحح الأخطاء القانونية أو اللغوية فقط عندما يكون
    التصحيح ممكنًا دون اختراع معلومات غير مدعومة.

11. إذا كانت الإجابة المقترحة تحتوي على معلومات صحيحة
    لكنها لا تجيب عن السؤال بالكامل، فأعد صياغتها لتغطي
    ما يدعمه السياق من السؤال.

12. إذا كان السياق يدعم جزءًا من الإجابة فقط، فاذكر الجزء
    المدعوم وحدد بإيجاز ما لا يمكن التحقق منه من السياق.

13. إذا لم يكن السياق كافيًا للإجابة، فأخرج العبارة التالية
    بحسب لغة السؤال:

    بالعربية:
    "{insufficient_context}"

    بالإنجليزية:
    "The retrieved legal context does not contain enough
    information to answer this question."

14. لا تخمّن الإجابة عند نقص المعلومات، ولا تستكملها
    اعتمادًا على الذاكرة أو المعرفة العامة.

15. اكتب إجابة واحدة مباشرة ومختصرة، دون تكرار المعنى
    أو تقديم ترجمة بلغة أخرى.

16. لا تعرض خطوات التفكير الداخلية أو تفاصيل المراجعة.

تعليمات الإخراج النهائي:
- أخرج نص الإجابة النهائية فقط.
- لا تكتب مقدمة أو تعليقًا على المراجعة.
- لا تستخدم عبارات مثل "الإجابة النهائية" أو
  "بعد المراجعة".
- التزم بلغة السؤال فقط.
- لا تكرر الإجابة باللغتين العربية والإنجليزية.
- لا تضف أي معلومة لا يدعمها السياق.

السياق القانوني المسترجع:
<context>
{context}
</context>

السؤال:
<question>
{question}
</question>

الإجابة المقترحة للمراجعة فقط، وقد تحتوي على أخطاء:
<answer>
{answer}
</answer>

أخرج الإجابة النهائية باللغة المطلوبة فقط:
""".strip()

