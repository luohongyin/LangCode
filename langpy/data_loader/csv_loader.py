from .loader import Loader


class CsvLoader(Loader):
    def __init__(self, data=None, file_name=None, read_mode='r'):
        super().__init__(data, file_name, read_mode)
        if file_name is None:
            raise ValueError('CsvLoader must process a file. set file_name.')

    def file_prompt(self, num_row, content_str):
        f_prompt = f"""# First {num_row} rows of the input file:\nexample_data = '''\n{content_str}\n ...\n'''\nfile_name = '{self.file_name}' \ninput_file = open(file_name)"""
        return f_prompt

    def get_prompt(self, notebook='jupyter', num_case=2):
        num_row = num_case + 1
        content_list = open(self.file_name, self.read_mode).readlines()[
            :num_row]
        content_list = [x.strip() for x in content_list]
        content_str = '\n# '.join(content_list)

        prompt = self.file_prompt(num_row, content_str)
        return self.format_prompt(prompt, notebook)
