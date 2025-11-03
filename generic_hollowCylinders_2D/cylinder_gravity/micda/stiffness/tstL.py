def tst_lambda_funs(x, y,z):
    def fl1(a, b):
        return a + b
    def fl2(a, b, c):
        return a * b * c
    nll1 = lambda *args: fl1(*args)
    nll2 = lambda *args: fl2(*args)
    nlls = lambda *args: fl1(*args) + fl2(*args)

    o1 = nll1(x,y)
    o2 = nll2(x,y,z)
    oo = nlls(x,y,z)
    exp1 = fl1(x,y)
    exp2 = fl2(x,y,z)
    exps = fl1(x,y) + fl2(x,y,z)

    print(exp1, o1)
    print(exp2, o2)
    print(exps, oo)
