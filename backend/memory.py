from compression import compress_text

class MemoryManager:
    def __init__(self):
        self.short_term = []
        self.long_term = []

    def add_message(self, message):
        self.short_term.append(message)

        if len(self.short_term) > 5:
            old = self.short_term.pop(0)
            compressed = compress_text(old, 200)
            self.long_term.append(compressed)

    def get_context(self):
        return " ".join(self.long_term + self.short_term)