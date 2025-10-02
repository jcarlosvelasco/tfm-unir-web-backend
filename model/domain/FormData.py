from pydantic import BaseModel


class FormData(BaseModel):
    cw: str
    edad: str
    sexo: str
    sa: str
    v: str
    dioptrias: str