from .data_loader.csv_loader import CsvLoader
from .data_loader.json_loader import JsonLoader

_API_ENDPOINT = 'https://lang-py-522564686dd7.herokuapp.com/items/0'
_SUPPORTED_PLATFORMS = {'gpt', 'palm'}


class LangPy:
    def __init__(self, api_key='', platform='gpt', model='gpt-4'):
        if not api_key:
            raise ValueError("Empty api_key")

        if not model:
            raise ValueError("Empty model")

        if platform not in _SUPPORTED_PLATFORMS:
            supported_platforms = ", ".join(_SUPPORTED_PLATFORMS)
            raise ValueError(
                "Non-supported platform. Supported ones: {supported_platforms}")

        self.api_key = api_key
        self.api_end_point = _API_ENDPOINT
        self.platform = platform
        self.model = model

    def get_http_prompt(self, http):
        prompt = ''
        if http == 'forbid':
            prompt = f'Do not send HTTP requests.'
        elif http == 'force':
            prompt = f'Send an HTTP request to solve the problem.'
        elif http == 'optional':
            pass
        else:
            raise ValueError(
                f'http mode: {http} not supported. try [forbid | force | optional]!')
        return prompt

    def get_data_loader_prompt(self,
                               notebook,
                               data=None,
                               file_name=None,
                               data_type='csv',  # csv or json
                               read_mode='r',
                               num_case=2):
        if data_type == 'csv':
            loader = CsvLoader(data, file_name, read_mode)
        elif data_type == 'json':
            loader = JsonLoader(data, file_name, read_mode)
        else:
            raise ValueError(
                f'data_type = {data_type} not supported. Please select from [csv | json]')
        return loader.get_prompt(notebook, num_case)

    def config_api_key(self):
        api_key = input('API key:')
        self.api_key = api_key

    def config_end_point(self):
        self.api_end_point = input('API endpoint:')

    def generate(self, instruction, hist_mode='all', http='optional'):
        pass

    def complete(self, instruction, hist_mode='all', http='optional'):
        pass

    def process_csv(self, file_name):
        pass
