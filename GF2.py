import numpy as np
# Irreducible polynomials from GF(2^2) to GF(2^100), e.g. [2, 1, 0] represents: x^2 + x^1 +x^0
IRREDUCIBLE_POLYs = [[2, 1, 0], [3, 1, 0], [4, 1, 0], [5, 2, 0], [6, 1, 0], [7, 1, 0], [8, 4, 3, 1, 0], [9, 1, 0],
                     [10, 3, 0], [11, 2, 0], [12, 3, 0], [13, 4, 3, 1, 0], [14, 5, 0], [15, 1, 0], [16, 5, 3, 1, 0],
                     [17, 3, 0], [18, 3, 0], [19, 5, 2, 1, 0], [20, 3, 0], [21, 2, 0], [22, 1, 0], [23, 5, 0],
                     [24, 7, 2, 1, 0], [25, 3, 0], [26, 4, 3, 1, 0], [27, 5, 2, 1, 0], [28, 1, 0], [29, 2, 0],
                     [30, 1, 0], [31, 3, 0], [32, 7, 3, 2, 0], [33, 6, 3, 1, 0], [34, 4, 3, 1, 0], [35, 2, 0],
                     [36, 5, 4, 2, 0], [37, 5, 4, 3, 2, 1, 0], [38, 6, 5, 1, 0], [39, 4, 0], [40, 5, 4, 3, 0],
                     [41, 3, 0], [42, 5, 2, 1, 0], [43, 6, 4, 3, 0], [44, 5, 0], [45, 4, 3, 1, 0], [46, 1, 0],
                     [47, 5, 0], [48, 5, 3, 2, 0], [49, 6, 5, 4, 0], [50, 4, 3, 2, 0], [51, 6, 3, 1, 0], [52, 3, 0],
                     [53, 6, 2, 1, 0], [54, 6, 5, 4, 3, 2, 0], [55, 6, 2, 1, 0], [56, 7, 4, 2, 0], [57, 4, 0],
                     [58, 6, 5, 1, 0], [59, 6, 5, 4, 3, 1, 0], [60, 1, 0], [61, 5, 2, 1, 0], [62, 6, 5, 3, 0],
                     [63, 1, 0], [64, 4, 3, 1, 0], [65, 4, 3, 1, 0], [66, 3, 0], [67, 5, 2, 1, 0], [68, 7, 5, 1, 0],
                     [69, 6, 5, 2, 0], [70, 5, 3, 1, 0], [71, 5, 3, 1, 0], [72, 6, 4, 3, 2, 1, 0], [73, 4, 3, 2, 0],
                     [74, 6, 2, 1, 0], [75, 6, 3, 1, 0], [76, 5, 4, 2, 0], [77, 6, 5, 2, 0], [78, 6, 4, 3, 2, 1, 0],
                     [79, 4, 3, 2, 0], [80, 7, 5, 3, 2, 1, 0], [81, 4, 0], [82, 7, 6, 4, 2, 1, 0], [83, 7, 4, 2, 0],
                     [84, 5, 0], [85, 8, 2, 1, 0], [86, 6, 5, 2, 0], [87, 7, 5, 1, 0], [88, 5, 4, 3, 2, 1, 0],
                     [89, 6, 5, 3, 0], [90, 5, 3, 2, 0], [91, 7, 6, 5, 3, 2, 0], [92, 6, 5, 2, 0], [93, 2, 0],
                     [94, 6, 5, 1, 0], [95, 6, 5, 4, 2, 1, 0], [96, 6, 5, 3, 2, 1, 0], [97, 6, 0], [98, 7, 4, 3, 0],
                     [99, 6, 3, 1, 0], [100, 6, 5, 2, 0]]


class GF2:
    """
    Build GF(2^m)
    """
    def __init__(self, m):
        self.m = m
        self.irreduciblePoly = np.zeros(m+1)
        for i in IRREDUCIBLE_POLYs[m-2]:
            self.irreduciblePoly[m-i] = 1
        self.irreduciblePoly = self.irreduciblePoly.astype(int)

    def reduce(self, x):
        xc = x.copy()
        if xc.shape[0] == 0:
            return np.array([0])
        if np.array_equal(xc, [0]):
            return xc
        if xc[0] != 0:
            return xc
        i = 0
        while i < xc.shape[0] and xc[i] == 0 :
            i += 1
        if i == xc.shape[0]:
            return np.array([0])
        else:
            return xc[i:]

    def polyMul(self, a, b):
        """
        Do polynomial multiplication
        :param a Coefficients of polynomial, x^n..x^0  NumPy array
        :param b Coefficients of polynomial, x^n..x^0  NumPy array
        :return a*b NumPy array
        """
        x = a.copy()
        y = b.copy()
        if np.array_equal(x, [1]):
            return y
        if np.array_equal(b, [1]):
            return x
        res = np.zeros(x.shape[0] + y.shape[0]).astype(int)
        for i in range(y.shape[0] - 1, -1, -1):
            if y[i] == 1:
                zeros = np.zeros([y.shape[0]-1 - i]).astype(int)
                tmp = np.concatenate((x, zeros))
                res = self.polySum(res, tmp)
        res = self.reduce(res)
        res = self.polyDiv(res, self.irreduciblePoly)

        return res

    def polySum(self, a, b):
        x = a.copy()
        y = b.copy()
        if a.shape[0] > b.shape[0]:
            zeros = np.zeros(a.shape[0]-b.shape[0]).astype(int)
            y = np.concatenate((zeros, b))
        elif b.shape[0] > a.shape[0]:
            zeros = np.zeros(b.shape[0] - a.shape[0]).astype(int)
            x = np.concatenate((zeros, a))

        for i in range(a.shape[0]):
            x[i] = (x[i] + y[i]) % 2
        return x

    def polyPower(self, a, n: int):
        """
        Do polynomial multiplication
        :param a Coefficients of polynomial, x^n..x^0  NumPy array
        :param n power
        :return a^n NumPy array
        """
        x = a.copy()
        if n == 0:
            return np.array([1])
        elif n == 1:
            return x
        i = 1
        res = x
        while 2*i < n:
            res = self.polyMul(res, res)
            res = self.polyDiv(res, self.irreduciblePoly)
            i *= 2
        res = self.polyMul(res, self.polyPower(x, n-i))
        return res

    def polyDiv(self, a, b):
        """
        Do polynomial multiplication
        :param a Coefficients of polynomial, x^n..x^0  NumPy array
        :param b Coefficients of polynomial, x^n..x^0  NumPy array
        :return a/b remainder, NumPy array
        """
        x = a.copy()
        y = b.copy()
        if x.shape[0] < y.shape[0]:
            return x
        elif x.shape[0] == y.shape[0]:
            return self.polySum(x, y)
        else:
            tmp = np.concatenate((y, np.zeros(x.shape[0] - y.shape[0]))).astype(int)
            tmp = self.polySum(tmp, x)
            tmp = self.reduce(tmp)
            return self.reduce(self.polyDiv(tmp, y))

    def findGenerator(self):
        for i in range(2, 2**self.m):
            alpha = np.asarray([int(x) for x in bin(i)[2:]])
            rem = self.polyPower(alpha, 2**self.m-1)
            if np.array_equal(rem, [1]):  # alpha^(2^m-1) = 1
                flag = 1
                power = alpha
                for j in range(1, 2**self.m-2):
                    power = self.polyMul(power, alpha)
                    if np.array_equal(power, [1]):  # alpha^(2^m-1) != 1
                        flag = 0
                        break
                if flag == 1:
                    return alpha


if __name__ == '__main__':
    ff = GF2(13)
    x = np.array([1,1,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,1,0,1,0,1])
    y = np.array([1,1,0,1])
    x = x.astype(int)
    y = y.astype(int)
    print(ff.irreduciblePoly)
    print(ff.polySum(y,x))
    print(ff.polyMul(y,x))
    print(ff.polyDiv(x,y))
    print(ff.polyPower(x,3))
