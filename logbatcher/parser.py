import json
import re
import string
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential
from logbatcher.cluster import Cluster
from logbatcher.postprocess import post_process
from logbatcher.sample import nearest_k_pairs_from_log
from logbatcher.matching import extract_variables, matches_template, prune_from_cluster
from logbatcher.postprocess import correct_single_template_full, correct_single_template
from logbatcher.util import verify_template, not_varibility

# Only import OpenAI and Together if we are not using mistral
# but we will handle that logic dynamically.
try:
    from openai import OpenAI
    from together import Together
except ImportError:
    OpenAI = None
    Together = None


class Parser:
    def __init__(self, model, theme, config):
        """
        Initialize the Parser class.
        :param model: Model name to use (e.g., 'mistral', 'gpt-3.5-turbo', etc.).
        :param theme: Theme for parsing logs.
        :param config: Configuration dictionary with API settings.
        """
        self.model = model
        self.theme = theme
        self.time_consumption_llm = 0
        self.use_mistral = "mistral" in self.model.lower()

        if self.use_mistral:
            # Using Mistral with Ollama
            self.api_url = config.get("mistral_api_url", "http://127.0.0.1:11434")
        else:
            # Using OpenAI or Together
            if config['api_key_from_openai'] == '<OpenAI_API_KEY>' and config['api_key_from_together'] == '<Together_API_KEY>':
                raise ValueError("Please provide your OpenAI and Together API keys in the config.json file.")

            if 'gpt' in self.model:
                # OpenAI API
                self.api_key = config['api_key_from_openai']
                self.client = OpenAI(api_key=self.api_key)
            else:
                # Together API
                self.api_key = config['api_key_from_together']
                self.client = Together(api_key=self.api_key)

    @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(10))
    def chat(self, messages):
        """
        Send messages to the LLM in a conversational style.
        :param messages: List of dict with 'role' and 'content'.
        :return: Model response as a string.
        """
        if self.use_mistral:
            # Mistral via Ollama
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            return self._send_mistral_request(prompt)
        else:
            # OpenAI or Together
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip('\n')

    @retry(wait=wait_random_exponential(min=1, max=8), stop=stop_after_attempt(10))
    def inference(self, prompt):
        """
        Send a single prompt to the LLM.
        :param prompt: The prompt string.
        :return: Model response as a string.
        """
        if self.use_mistral:
            # Mistral via Ollama
            return self._send_mistral_request(prompt)
        else:
            # OpenAI or Together
            retry_times = 0
            output = ''
            while True:
                try:
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=0.0,
                    )
                    output = response.choices[0].text.strip('\n')
                except Exception as e:
                    print(e)
                    retry_times += 1
                    if retry_times > 3:
                        return output
                else:
                    return output

    def _send_mistral_request(self, prompt):
        """
        Send a request to the Mistral model via Ollama.
        :param prompt: The prompt string.
        :return: Response text from the model.
        """
        url = f"{self.api_url}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {"model": self.model, "prompt": prompt, "temperature": 0.0}

        with httpx.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()  # Raise exception for HTTP errors
            result = ""
            for line in response.iter_text():
                if line.strip():
                    chunk = json.loads(line)
                    result += chunk.get("response", "")
                    if chunk.get("done", False):
                        break
            return result.strip()

    def get_responce(self, cluster, cached_pairs={}, sample_pairs=[], shot=0, dataset='Apache', data_type='2k'):
        """
        Process log clusters to extract templates using the model (Mistral or OpenAI/Together).
        :param cluster: Log cluster to process.
        :param cached_pairs: Cached template-log pairs for optimization.
        :param sample_pairs: Sample template-log pairs for few-shot prompting.
        :param shot: Number of shots (examples) for few-shot prompting.
        :param dataset: Dataset name.
        :param data_type: Data type (e.g., '2k', 'full').
        :return: Extracted template, updated cluster, and pruned new cluster.
        """
        logs = cluster.batch_logs
        sample_log = logs[0]
        if type(logs) == str:
            logs = [logs]

        if not_varibility(logs) and data_type == 'full':
            print("no varibility")
            logs = [f'{sample_log}']
            logs = [sample_log]

        new_cluster = Cluster()
        # caching
        for template, referlog_and_freq in cached_pairs.items():
            for log in cluster.logs:
                match_result = matches_template(log, [referlog_and_freq[0], template])
                if match_result is not None:
                    cluster, new_cluster = prune_from_cluster(template, cluster)
                    cached_pairs[template][1] += len(new_cluster.logs)
                    return match_result, cluster, new_cluster

        demonstrations = ''
        # using labelled data
        if shot > 0:
            nearest_k_pairs = nearest_k_pairs_from_log(sample_log, sample_pairs, shot)
            for i in range(shot):
                demonstrations += f"Log message: {nearest_k_pairs[shot - i - 1][0]}\nLog template: {nearest_k_pairs[shot - i - 1][1].replace('<*>', '{{variable}}')}\n"

        instruction = "You will be provided with some log messages separated by line break. You must abstract variables with {{placeholders}} to extract the corresponding template. There might be no variables in the log message.\nPrint the input log's template delimited by backticks."

        if demonstrations != '':
            query = demonstrations + 'Log message:\n' + '\n'.join([f'{log}' for log in logs]) + '\nLog template: '
        elif all(model_type not in self.model for model_type in ['gpt', 'instruct', 'chat']):
            query = 'Log message:\n' + '\n'.join([f'{log}' for log in logs]) + '\nLog template: '
        else:
            query = '\n'.join(logs)

        # invoke LLM
        if any(model_type in self.model for model_type in ['gpt', 'instruct', 'chat']):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query}
            ]

            try:
                t0 = time.time()
                answer = self.chat(messages)
                self.time_consumption_llm += (time.time() - t0)
            except Exception as e:
                print("invoke LLM error", e)
                answer = sample_log
        else:
            prompt = f"{instruction}\n{query}"
            answer = self.inference(prompt)

        template = post_process(answer, data_type)
        if not verify_template(template):
            if data_type == 'full':
                template = correct_single_template_full(sample_log)
            else:
                template = correct_single_template(sample_log)

        # matching and pruning
        for log in logs:
            try:
                matches = extract_variables(log, template)
            except:
                matches = None
            if matches is not None:
                parts = template.split('<*>')
                template = parts[0]
                for index, match in enumerate(matches):
                    if match != '':
                        template += '<*>'
                    template += parts[index + 1]
                break
        else:
            if data_type == 'full':
                template = correct_single_template_full(sample_log)
            else:
                template = correct_single_template(sample_log)

        cluster, new_cluster = prune_from_cluster(template, cluster)
        return template, cluster, new_cluster
