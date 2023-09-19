import IPython
from IPython.core.display import display, Javascript
from .base import LangPy


class JupyterLangPy(LangPy):
    def __init__(self, api_key='', platform='gpt', model='gpt-4'):
        super().__init__(api_key=api_key, platform=platform, model=model)

    def get_prompt(self, instruction, http):
        http_prompt = self.get_http_prompt(http)
        return f'{instruction} {http_prompt}'

    def init_js_code(self):
        return """
        async function myCallbackFunction(data) {
            const url = '%s';
            try {
                let response = await fetch(url, {
                    method: "PUT",
                    headers: {
                    "Content-Type": "application/json",
                    },
                    body: JSON.stringify(data),
                });
                let responseData = await response.json();
                return responseData;
            } catch (error) {
                console.log(error);
            }
        }
        var current_index = Jupyter.notebook.get_selected_index();
        var previous_cell_content = "";
        """ % self.api_end_point

    def js_get_history(self, hist_mode):
        if hist_mode == "all":
            js_code = """
        for (var i = 1; i < current_index; i++) {
            var cell = Jupyter.notebook.get_cell(i);
            if (cell.cell_type === "code") {
                previous_cell_content += '```\\n' + cell.get_text() + '\\n```\\n';
            } else {
                previous_cell_content += cell.get_text() + '\\n';
            }
        }
        """
        elif isinstance(hist_mode, int):
            js_code = """
        for (var i = current_index - %d; i < current_index; i++) {
            var cell = Jupyter.notebook.get_cell(i);
            if (cell.cell_type === "code") {
                previous_cell_content += '```\\n' + cell.get_text() + '\\n```\\n';
            } else {
                previous_cell_content += cell.get_text() + '\\n';
            }
        }
        """ % hist_mode
        elif hist_mode == 'single':
            js_code = ''
        else:
            raise ValueError(
                f"hist_mode {hist_mode} not supported. Try [all | single | INT]")

        return js_code

    def build_request_data(self, instruction, exist_code=False):
        if exist_code:
            js_request = """
            previous_cell_content += '%s'
            var api_key_str = "%s"
            var next_cell = Jupyter.notebook.get_cell(current_index)
            var exist_code = next_cell.get_text()

            var data = {
                "instruction": previous_cell_content,
                "api_key_str": api_key_str,
                "exist_code": exist_code,
                "platform": "%s",
                "model": "%s"
            }
            """ % (instruction, self.api_key, self.platform, self.model)
        else:
            js_request = """
            previous_cell_content += '%s'
            var api_key_str = "%s"
            var data = {
                "instruction": previous_cell_content,
                "api_key_str": api_key_str,
                "exist_code": "none",
                "platform": "%s",
                "model": "%s"
            }
            """ % (instruction, self.api_key, self.platform, self.model)
        return js_request

    def build_js_code(self, instruction, hist_mode, http, exist_code=False):
        instruction = self.get_prompt(instruction, http)

        js_code = self.init_js_code()
        js_code += self.js_get_history(hist_mode)
        js_code += self.build_request_data(instruction, exist_code=exist_code)
        return js_code

    def generate(self, instruction, hist_mode='single', http='optional'):
        js_code = self.build_js_code(
            instruction, hist_mode, http, exist_code=False
        )

        js_code += """
        myCallbackFunction(data).then(responseData => {
            var print_content = responseData["output"]
            Jupyter.notebook.select_prev()
            var new_cell = Jupyter.notebook.insert_cell_below('code');

            var next_cell = Jupyter.notebook.get_cell(current_index)
            next_cell.set_text(print_content);
        }).catch()
        """

        print('The code will be generated in the next cell ...')

        display(Javascript(js_code))

    def complete(self, instruction, hist_mode='single', http='optional'):
        js_code = self.build_js_code(
            instruction, hist_mode, http, exist_code=True
        )

        js_code += """
        myCallbackFunction(data).then(responseData => {
            var print_content = responseData["output"]
            var next_cell = Jupyter.notebook.get_cell(current_index)
            next_cell.set_text(print_content);
        })
        """
        print('The code in the following cell will be completed ...')

        display(Javascript(js_code))

    def preview_data(
            self,
            data=None,
            file_name=None,
            data_type='csv',  # csv or json
            read_mode='r',
            num_case=2):
        prompt = self.get_data_loader_prompt(
            'jupyter', data, file_name, data_type, read_mode, num_case)

        js_code = f"""
            Jupyter.notebook.select_prev()
            var new_cell = Jupyter.notebook.insert_cell_below('code');
            new_cell.set_text(`{prompt}`);
        """
        print('Run the following cell to read process the format of your csv file.')

        display(Javascript(js_code))
