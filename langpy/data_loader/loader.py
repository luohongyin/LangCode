class Loader:
    def __init__(self, data=None, file_name=None, read_mode='r'):
        self.data = data
        self.file_name = file_name
        self.read_mode = read_mode

    def format_prompt(self, prompt, notebook):
        if notebook == 'jupyter':
            prompt = prompt.replace('\n', '\\n').replace('\t', '\\t')
        elif notebook == 'colab':
            pass
        else:
            raise ValueError(
                f'Notebook = {notebook} note supported. Use [jupyter | colab]')
        return prompt