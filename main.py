# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def part1():
    import numpy    # import numpy library to use FFT and arrays
    from numpy import random    # import numpy random function
    import time  # time()

    # Q1.1
    x = random.rand(65536)
    y = random.rand(65213)
    tic = time.time()  # count start time
    x1 = numpy.fft.fft(x)
    toc = time.time()  # count finish time
    print('FFT(x)=', (toc-tic))  # time elapse
    xtime=(toc-tic)

    tic = time.time()  # count start time
    y1 = numpy.fft.fft(y)
    toc = time.time()  # count finish time
    print('FFT(y)=', (toc-tic))  # time elapsed
    ytime=(toc-tic)

    timedifference=numpy.abs(ytime-xtime)
    print('FFT(|x-y|)=', timedifference)

def part1_2():
    import numpy as np
    np.set_printoptions(precision=2, linewidth=100)

    # Q1.2, find 8-point FFT of 4 signals
    x1 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    x2 = np.array([1, 1, -1, 0, 1, 0, -1, 1])
    x3 = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    x4 = np.array([0, 1, 1, 1, 0, -1, -1, -1])

    h1 = np.fft.fft(x1, 8)
    h2 = np.fft.fft(x2, 8)
    h3 = np.fft.fft(x3, 8)
    h4 = np.fft.fft(x4, 8)

    print('\n', h1, '\n\n', h2, '\n\n', h3, '\n\n', h4)

def part1_3():
    import numpy as np
    from matplotlib import pyplot as plot
    # np.set_printoptions(precision=2, linewidth=100)

    # Q1.3
    x1 = [1, 1, 1, 1]

    x2 = np.pad(x1, (0, 4), 'constant')
    x3 = np.pad(x1, (0, 28), 'constant')
    x4 = np.pad(x1, (0, 124), 'constant')
    x5 = np.pad(x1, (0, 1020), 'constant')

    y1 = np.fft.fft(x1)
    y2 = np.fft.fft(x2, 8)
    y3 = np.fft.fft(x3, 32)
    y4 = np.fft.fft(x4, 128)
    y5 = np.fft.fft(x5, 1024)

    magn_x1 = abs(y1)
    magn_x2 = abs(y2)
    magn_x3 = abs(y3)
    magn_x4 = abs(y4)
    magn_x5 = abs(y5)

    plot.stem(magn_x5)
    plot.xlim([-1, 1024])  #[-1, x] where x = n of np.fft.fft(xn, n)
    plot.ylim([-1, 5])
    plot.show()

def part2_1():
    import numpy as np

    # Q2.1
    x = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 1]
    h = [1, 4, 6, 4, 1]
    y = np.convolve(x,h)
    print(y)

def part2_2():
    import numpy as np
    from sklearn.metrics import mean_squared_error as mse
    np.set_printoptions(precision=1, linewidth=300)

    # Q2.2 Show that linear convolution in time = point-by-point multiplication in frequency (FFT)
    # x = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 1]
    x = [1, 2, 3, 4, 5, 6, 5, 4, 3]
    h = [1, 4, 6, 4, 1]

    # Direct convolution
    y = np.convolve(x, h)
    print('\nDirect Convolution(t): ', y)

    # Point-by-point multiplication: pad, fft, pbp, ifft
    x = np.pad(x, (0, 3), 'constant')  # pad x
    # xh = np.pad(x, (0, 3), 'constant') pad h
    x01 = np.fft.fft(x, 16)  # fft xp
    h01 = np.fft.fft(h, 16)  # fft hp
    y01 = abs(np.multiply(x01, h01))  # point-by-point multiplication
    y02 = np.fft.fft(y)  # fft of direct convolution
    print('Direct Convolution(w): ', y02)

    y03 = np.fft.ifft(y01)
    print('FFT(t): ', y03)
    print('FFT(t) clean: ', np.real(y03))
    print('FFT(w): ', y01)

    # diff = mse(abs(y02), abs(y01))

def part2_3():
    import numpy as np
    from numpy import random
    import time

    x = random.rand(10500)
    h = random.rand(1024)

    tic = time.time()
    y = np.convolve(x, h)
    toc = time.time()
    print('Direct Convolution: ', y)
    print('time: ', toc - tic)
    t = toc - tic

    tic = time.time()
    x = np.pad(x, (0, 16384-10500), 'constant')
    h = np.pad(h, (0, 16384-1024), 'constant')
    x01 = np.fft.fft(x, 16384)
    h01 = np.fft.fft(h, 16384)
    y01 = abs(np.multiply(x01, h01))
    y03 = np.fft.ifft(y01)
    toc = time.time()

    print('FFT: ', y)
    print('time: ', toc - tic)
    print('time saved: ', np.abs(t - (toc - tic)))

if __name__ == '__main__':
    # part1()
    # part1_2()
    # part1_3()
    # part2_1()
    # part2_2()
    part2_3()