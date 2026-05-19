'''

用于将emotion映射到VA向量

'''


def getVec2(n):
    return ((-0.6, 0.6), (-0.7, 0.2), (-0.5, 0.8), (0.8, 0.2), (-0.8, -0.4), (0.2, 0.8), (0.0, 0.0))[n]

if __name__=="__main__":
    for i in range(0,7):
        print(getVec2(i))