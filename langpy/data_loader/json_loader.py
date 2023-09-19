import json
from .loader import Loader


class JsonLoader(Loader):
    def __init__(self, data=None, file_name=None, read_mode='r'):
        super().__init__(data, file_name, read_mode)
        if data is None and file_name is None:
            raise ValueError('Must provide either `data` or `file_name`.')

    def file_prompt(self, num_case, content_str):
        f_prompt = f"""# First {num_case} cases of the Json data:\nexample_data = '''\n{content_str}\n...\n'''\nfile_name = '{self.file_name}' \ninput_file = open(file_name)"""
        return f_prompt

    def data_prompt(self, num_case, content_str):
        d_prompt = f"""# First {num_case} cases of the Json data:\nexample_data = '''\n{content_str}\n...\n'''"""
        return d_prompt

    def get_prompt(self, notebook='jupyter', num_case=2):
        if self.data is None:
            self.data = json.load(open(self.file_name, self.read_mode))

        example_list = self.data[:num_case]
        content_str = ',\n'.join([json.dumps(x, indent=4)
                                 for x in example_list])
        
        if self.file_name is not None:
            prompt = self.file_prompt(num_case, content_str)
        else:
            prompt = self.data_prompt(num_case, content_str) 

        return self.format_prompt(prompt, notebook)
