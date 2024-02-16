class ExperimentProgressLogger():
    def __init__(self):
        self.progress = None
        self.message = None

    def log(self, event):
        self.progress = event.progress
        self.message = event.message
        print(self.progress, self.message)

    def get_formatted_log(self):
        return f"Progress: {self.progress * 100:.2f}%. {self.message}"