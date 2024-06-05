import argparse
import json
import logging
import random
import requests
import re
import vertexai
from vertexai.preview.generative_models import grounding
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import generative_models
import time

logger = logging.getLogger(__name__)


def get_questions(num_questions):
    """
    Get questions from OpenTrivia and filter out unnecessary fields
    """

    api_url = "https://opentdb.com/api.php"
    params = {
        'amount': num_questions,
        'type': 'multiple'
    }

    response = requests.get(api_url, params=params)

    if response.status_code != 200:
        logger.error(f"Failed to fetch data from Open Trivia DB. Status code: {response.status_code}")
        return

    data = response.json()
    if data['response_code'] != 0:
        logger.error("Open Trivia DB API returned an error (code:", data['response_code'], ")")
        return

    filtered_questions = [
        {
            'question': result['question'],
            'correct_answer': result['correct_answer'],
            'incorrect_answers': result['incorrect_answers']
        }
        for result in data['results']
    ]

    return filtered_questions


def transform_questions(questions):
    """
    Combine incorrect_answers and correct_answers into a single answers field (to hide the correct answer from LLM)
    """
    return [
        {
            'question': question['question'],
            'answers': random.sample(question['incorrect_answers'] + [question['correct_answer']], 4)
        }
        for question in questions
    ]


def ask_llm(project_id, model_name, google_search_grounding, questions_transformed):
    """
    Pass the transformed questions to LLM and ask it to find the correct answer
    """
    vertexai.init(project=project_id, location="us-central1")

    model = GenerativeModel(model_name=model_name, generation_config={"response_mime_type": "application/json"})

    prompt = f"""Given these questions and answers in JSON, can you find the correct answer?
Make sure you preserve the JSON structure in your output
Add 1 correct answer as "correct_answer"
Add the rest of 3 answers as "incorrect_answers"
Remove "answers" field
Here's the JSON: {json.dumps(questions_transformed, indent=2)}
    """

    logger.debug(f"Prompt: {prompt}")

    tools = [Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())] if google_search_grounding else None

    response = model.generate_content(prompt,
                                      generation_config={ "temperature": 0},
                                      safety_settings={
                                            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE},
                                    tools=tools
    )

    logger.debug(f"Response.text: {response.text}")
    sanitized_text = remove_json_markers(response.text)
    return json.loads(sanitized_text)


def remove_json_markers(json_string, start_marker="```json", end_marker="```"):
    """
    Sometimes LLM wraps the response into ```json``` markers. Remove them.
    """

    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    match = re.search(pattern, json_string, re.DOTALL)  # re.DOTALL to match newlines

    if match:
        logger.debug(f"Removed JSON marker")
        return match.group(1)
    return json_string


def compare_question_lists(questions_original, questions_graded):
    """
    Given original questions and LLM graded questions, see how many correct answers match
    """
    count_correct = 0
    for original_question in questions_original:
        found_match = False
        for graded_question in questions_graded:
            if original_question['question'] == graded_question['question']:
                if original_question['correct_answer'] == graded_question['correct_answer']:
                    count_correct += 1
                else:
                    logger.debug(f"Mismatch for question: '{original_question['question']}'")
                    logger.debug(f"  Original correct answer: '{original_question['correct_answer']}'")
                    logger.debug(f"  Graded correct answer: '{graded_question['correct_answer']}'")
                found_match = True
                break  # Move to the next original question after finding a match

        if not found_match:
            logger.debug(f"Question not found in graded list: '{original_question['question']}'")

    logger.info(f"Questions with matching correct answers: {count_correct}")

    percentage_correct = (count_correct / len(questions_original)) * 100
    return percentage_correct


def run_test(project_id, model_name, no_questions, google_search_grounding):
    start_time = time.time()

    logger.info(f"Questions requested: {no_questions}")

    questions_original = get_questions(no_questions)
    logger.debug("questions_original: " + json.dumps(questions_original, indent=2))

    questions_transformed = transform_questions(questions_original)
    logger.debug("questions_transformed: " + json.dumps(questions_transformed, indent=2))

    questions_graded = ask_llm(project_id, model_name, google_search_grounding, questions_transformed)
    logger.debug(f"questions_graded: " + json.dumps(questions_graded, indent=2))

    percentage_correct = compare_question_lists(questions_original, questions_graded)
    logger.info(f"Percentage correct: {percentage_correct:.2f}%")

    execution_time = time.time() - start_time
    logger.info(f"Execution time: {execution_time:.2f} seconds")

    return percentage_correct, execution_time


def run_tests(project_id, model_name, num_iterations, no_questions, google_search_grounding):
    logger.info(f"=============================")
    logger.info(f"Project: {project_id}")
    logger.info(f"No of iterations: {num_iterations}")
    logger.info(f"No of questions per iteration: {no_questions}")
    logger.info(f"Google search grounding: {google_search_grounding}")
    logger.info(f"Model: {model_name}")
    logger.info(f"=============================")

    results = []
    execution_times = []

    for iteration in range(num_iterations):
        logger.info(f"== Test run: {iteration + 1} ==")
        percentage_correct, execution_time = run_test(project_id, model_name, no_questions, google_search_grounding)
        results.append(percentage_correct)
        execution_times.append(execution_time)

    average_percentage = sum(results) / num_iterations
    min_percentage = min(results)
    max_percentage = max(results)

    average_execution_time = sum(execution_times) / num_iterations
    min_execution_time = min(execution_times)
    max_execution_time = max(execution_times)

    logger.info(f"=============================")
    logger.info(f"Percentage correct: min={min_percentage:.2f}%, avg={average_percentage:.2f}%, max={max_percentage:.2f}% | "
                f"Execution time: min={min_execution_time:.2f}s, avg={average_execution_time:.2f}s, max={max_execution_time:.2f}s")
    logger.info(f"=============================\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Process project and model information.')
    parser.add_argument('project_id', type=str, help='Google Cloud project id')
    parser.add_argument('model_name', type=str, help='Model name')
    parser.add_argument('--num_iterations', type=int, default=4, help='Number of iterations (default: 4)')
    parser.add_argument('--no_questions', type=int, default=25, help='Number of questions per iteration (default: 25)')
    parser.add_argument('--google_search_grounding', action='store_true', help='Use Google search grounding (default: False)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    run_tests(args.project_id, args.model_name, args.num_iterations, args.no_questions, args.google_search_grounding)


