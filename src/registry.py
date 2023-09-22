class Registry:

    def __init__(self):
        self.map = {}

    def keys(self):
        return self.map.keys()

    def __getitem__(self, name):
        return self.map[name]

    def register(self, name):
        return self.Register(self, name)

    class Register:
        def __init__(self, outer_self, name):
            self.name = name
            self.outer_self = outer_self
        
        def __call__(self, fn):
            self.outer_self.map[self.name] = fn
            return fn