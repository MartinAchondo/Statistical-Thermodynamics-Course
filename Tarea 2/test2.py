
class Test():

    def __init__(self):
        self.x = 10


    def __del__(self):
        self.x = 5
        print(self.x)


a = Test()

print(a.x,'saksxas')

del a

a.x
