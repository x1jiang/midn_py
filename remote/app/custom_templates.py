from fastapi.templating import Jinja2Templates
import json

# Create a custom Jinja2Templates class with tojson filter
class CustomTemplates(Jinja2Templates):
    def __init__(self, directory):
        super().__init__(directory=directory)
        self.env.filters["tojson"] = lambda obj: json.dumps(obj)

# Replace the standard templates with our custom one
templates = CustomTemplates(directory="remote/app/static")
