import os

class FileReloader():
    def __init__(self, path, loader_func):
        self.path = path
        self.last_modification = 0
        self.loader_func = loader_func
        self.data = None

    def was_modified(self):
        '''Check if the file was modified since last reload'''
        if os.path.getmtime(self.path) > self.last_modification:
            return True
        return False
    
    def reload(self):
        '''Reload the newer data or return the cached data'''
        if self.was_modified():
            self.last_modification = os.path.getmtime(self.path)
            self.data = self.loader_func(self.path)
        return self.data