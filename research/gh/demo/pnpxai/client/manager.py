from .database import Database

class ObjectManager:
    def __init__(self, cls, parent=None):
        self.cls = cls
        self.parent = parent
        
    def create(self, **kwargs):
        with Database(table_name=self.cls._table_name) as table:
            obj = self.cls(**kwargs)
            record = table.insert(obj.to_dict())
        return self.get(**record)

    def get(self, **kwargs):
        with Database(table_name=self.cls._table_name) as table:
            records = table.select(**kwargs)
        assert len(records) <= 1, "Not unique"
        if len(records) == 0:
            return None
        return self.cls(**records[0])

    def get_or_create(self, **kwargs):
        obj = self.get(**kwargs)
        if obj:
            return obj, False
        obj = self.create(**kwargs)
        return obj, True

    def update(self, **kwargs):
        with Database(table_name=self.cls._table_name) as table:
            table.delete(kwargs["id"])
            table.insert(kwargs)
        return self.cls(**kwargs)
        
    def all(self):
        if self.parent:
            with Database(table_name=self.parent._relation_table_name(self.cls)) as table:
                records = table.select(**{self.parent._relation_id_key: self.parent.id})
                indices = [record[self.cls._relation_id_key] for record in records]
            return [self.get(id=i) for i in indices]
        with Database(table_name=self.cls._table_name) as table:
            records = table.select()
        return [self.cls(**record) for record in records]
        
    def filter(self, **kwargs):
        return

    def delete(self, id, **kwargs):
        return