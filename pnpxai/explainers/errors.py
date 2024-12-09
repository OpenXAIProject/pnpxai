class NoCamTargetLayerAndNotTraceableError(Exception):
    def __init__(self, *messages):
        self.messages = messages

    def __str__(self):
        return ' '.join(self.messages)

