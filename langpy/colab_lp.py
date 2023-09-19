import json
import requests
from .base import LangPy
from google.colab import _message


class ColabLangPy(LangPy):
    def __init__(self, api_key='', platform='gpt', model='gpt-4'):
        super().__init__(api_key=api_key, platform=platform, model=model)
        self._colab_list = []
        self._cur_idx = -1

    def format_cell(self, cell_tuple):
        cell_str, cell_type = cell_tuple
        if cell_type == 'code':
            return f'```\n{cell_str}\n```'
        else:
            return cell_str

    def post_code(self, code):
        _message.blocking_request(
            'add_scratch_cell',
            request={'content': code, 'openInRightPane': True},
            timeout_sec=None
        )

    def update_idx_hist(self, instruction):
        colab_hist = _message.blocking_request('get_ipynb')['ipynb']['cells']
        self._code_list = [
            [''.join(x['source']), x['cell_type']] for x in colab_hist
        ]

        self._code_list = [
            self.format_cell(ct) for ct in self._code_list
        ]

        self._cur_idx = -1
        for i, code in enumerate(self._code_list):
            if instruction in code:
                self._cur_idx = i
                break

    def get_history_prompt(self, instruction, hist_mode):
        prompt = ''
        if hist_mode == 'all':
            previous_code = '\n\n'.join(self._code_list[:self._cur_idx])
            prompt = f'{previous_code}\n\n{instruction}'
        elif hist_mode == 'single':
            prompt = instruction
        elif isinstance(hist_mode, int):
            previous_code = '\n\n'.join(
                self._code_list[self._cur_idx - hist_mode: self._cur_idx])
            prompt = f'{previous_code}\n\n{instruction}'
        else:
            raise ValueError(
                f'hist_mode = {hist_mode} not supported. try [all | single | INTEGER].')
        return prompt

    def get_prompt(self, instruction, hist_mode, http):
        history_prompt = self.get_history_prompt(instruction, hist_mode)
        http_prompt = self.get_http_prompt(http)
        return f'{history_prompt} {http_prompt}'

    def get_code_of_thought(self, json_data):
        response = requests.put(self.api_end_point, json=json_data)
        if response.status_code != 200:
            raise SystemError(
                f'Failed to get valid response from server: {response.status_code}')
        return response.content

    def generate(self, instruction, hist_mode='single', http='optional'):
        self.update_idx_hist(instruction)
        response = self.get_code_of_thought(
            json_data={
                'instruction': self.get_prompt(instruction, hist_mode, http),
                'api_key_str': self.api_key,
                'exist_code': 'none',
                'platform': self.platform,
                'model': self.model
            }
        )
        ans_str = json.loads(response)['output']
        self.post_code(ans_str)

    def complete(self, instruction, hist_mode='single', http='optional'):
        self.update_idx_hist(instruction)
        response = self.get_code_of_thought(
            json_data={
                'instruction': self.get_prompt(instruction, hist_mode, http),
                'api_key_str': self.api_key,
                'exist_code': self._code_list[self._cur_idx + 1].replace('```', '').strip(),
                'platform': self.platform,
                'model': self.model
            }
        )
        ans_str = json.loads(response)['output']
        self.post_code(ans_str)

    def preview_data(
            self,
            data=None,
            file_name=None,
            data_type='csv',  # csv or json
            read_mode='r',
            num_case=2):
        prompt = self.get_data_loader_prompt(
            'colab', data, file_name, data_type, read_mode, num_case)
        self.post_code(prompt)
