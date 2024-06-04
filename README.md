# OpenTrivia LLM testing

A test suite to see if LLMs can correctly guess the correct answer in OpenTrivia
quizzes.

Install dependencies:

```sh
pip install -r requirements.txt
```

Minimal run:

```sh
python main.py your-project-id model-id
```

Run with all options:

```sh
python main.py your-project-id model-id --num_iterations=4  --no_questions=25 --google_search_grounding
```

Examples:

```sh
python main.py genai-atamel gemini-1.0-pro-002
python main.py genai-atamel gemini-1.5-pro-001
python main.py genai-atamel gemini-1.5-flash-001
```
