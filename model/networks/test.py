def print1(*params):
    print(params)

def print2(**params):
    print(params)

# print1(1,2)
para = [1,2]
print(*para)
# print2(x=1,y=2,z=3)