from fastapi import Query, HTTPException


class Dep:
    def __init__(self, name:str = Query(...,max_length=10,min_length=2),num:int = Query(...,ge=1,le=10)):
        self.name = name
        self.num = num

