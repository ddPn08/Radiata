from pydantic import BaseModel

class BaseModelStream(BaseModel):
    def ndjson(self):
        return self.json() + '\n'