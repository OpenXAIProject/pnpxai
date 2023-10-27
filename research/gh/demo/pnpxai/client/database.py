import os
import urllib
import json

DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH = "./xaistore"

def _default_root_dir():
    return os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)

def mkdir(root, name):
    target = os.path.join(root, name) if name else root
    os.makedirs(target)
    return target

INPUT_OBJECT_TABLES = [
    {
        "name": "datasets",
        "fields": {"id": "int", "name": "str", "uri": "str", "origin": "str"},
        "data": []
    },
    {
        "name": "models",
        "fields": {"id": "int", "name": "str", "uri": "str", "origin": "str"},
        "data": []
    },
]
OUTPUT_OBJECT_TABLES = [
    {
        "name": "projects",
        "fields": {"id": "int", "name": "str"},
        "data": []
    },
    {
        "name": "tasks",
        "fields": {
            "id": "int",
            "name": "str",
            "project_id": "int",
            "dataset_id": "int",
            "model_id": "int",
        },
        "data": []
    },
    {
        "name": "explanations",
        "fields": {
            "id": "int",
            "name": "str",
            "task_id": "str",
            "explainer": "str",
        },
        "data": []
    },
]
RELATION_TABLES = [
    {"name": "project_id_to_dataset_id", "fields": {"id": "int", "project_id": "int", "dataset_id": "int"}, "data": []},
    {"name": "project_id_to_model_id", "fields": {"id": "int", "project_id": "int", "model_id": "int"}, "data": []},
    {"name": "project_id_to_task_id", "fields": {"id": "int", "project_id": "int", "task_id": "int"}, "data": []},
]


class Database:
    DEFAULT_DB_FILENAME = "db.json"
    
    def __init__(self, table_name, root_directory=None, db_filename=None):
        self.table_name = table_name
        self.root_directory = root_directory or _default_root_dir()
        self.db_filename = db_filename or self.DEFAULT_DB_FILENAME
        if not os.path.exists(self.root_directory):
            self._create_root_and_tables()

    @property
    def db_path(self):
        return os.path.join(self.root_directory, self.db_filename)

    def _create_root_and_tables(self):
        mkdir(self.root_directory, None)
        with open(os.path.join(self.root_directory, self.db_filename), 'w') as f:
            tables = INPUT_OBJECT_TABLES + OUTPUT_OBJECT_TABLES + RELATION_TABLES
            json.dump(tables, f)

    def __enter__(self):
        with open(self.db_path, "r") as f:
            self.db = json.load(f)
        self.table_index = self._get_table_index_by_name(self.table_name)
        self.fields = self.db[self.table_index]["fields"]
        self.data = self.db[self.table_index]["data"]
        return self

    def __exit__(self, type=None, value=None, traceback=None):
        pass

    def _get_table_index_by_name(self, table_name):
        return [i for i in range(len(self.db)) if self.db[i]["name"]==self.table_name][0]

    def _get_field_loc_by_name(self, field_name):
        return list(self.fields.keys()).index(field_name)

    def _next_id(self):
        if len(self.data) > 0:
            return max(d[0] for d in self.data) + 1
        return 0

    def _validate(self, new_record):
        # some validation process such as uniqueness
        return True

    def _ensure_order_and_type(self, record):
        return {nm: eval(tp)(record[nm]) for nm, tp in self.fields.items()}

    def _data_to_record(self, ordered_data):
        return dict(zip(self.fields.keys(), ordered_data))

    def save(self):
        with open(self.db_path, "w") as f:
            self.db[self.table_index]["data"] = self.data
            json.dump(self.db, f)

    def insert(self, new_record):
        if not new_record.get("id") != None:
            new_record["id"] = self._next_id()
        new_record = self._ensure_order_and_type(new_record)
        self.data.append(tuple(new_record.values()))
        self.save()
        return new_record

    def select(self, **kwargs):
        table = self.db[self.table_index].copy()
        if kwargs:
            selected = list()
            for d in self.data:
                # equal condition only
                if all(d[self._get_field_loc_by_name(k)] == v for k, v in kwargs.items()):
                    selected.append(self._data_to_record(d))
            return selected
        return [self._data_to_record(d) for d in self.data]

    def delete(self, idx):
        self.data = [d for d in self.data if d[0] != idx]
        self.save()