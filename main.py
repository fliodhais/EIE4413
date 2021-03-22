
def part1():
    import numpy
    from numpy import random
    import time  # time()

    # Q1.1 processing time comparison of fft(x) where x is a power of 2 and not.
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

    # Q1.3 plot 4/8/32/128/1024 point dft
    x1 = [1, 1, 1, 1]

    x2 = np.pad(x1, (0, 4), 'constant')
    x3 = np.pad(x1, (0, 28), 'constant')
    x4 = np.pad(x1, (0, 124), 'constant')
    x5 = np.pad(x1, (0, 1020), 'constant')
    x6 = np.pad(x1, (0, 12796), 'constant')

    y1 = np.fft.fft(x1)
    y2 = np.fft.fft(x2, 8)
    y3 = np.fft.fft(x3, 32)
    y4 = np.fft.fft(x4, 128)
    y5 = np.fft.fft(x5, 1024)
    y6 = np.fft.fft(x6, 12800)

    magn_x1 = abs(y1)
    magn_x2 = abs(y2)
    magn_x3 = abs(y3)
    magn_x4 = abs(y4)
    magn_x5 = abs(y5)
    magn_x6 = abs(y6)

    # plot.stem(magn_x6)
    # plot.xlim([899, 901])  #[-1, x] where x = n of np.fft.fft(xn, n)
    # plot.ylim([-1, 5])
    # plot.show()
    print(magn_x6[900])

def part2_1():
    import numpy as np

    # Q2.1
    x = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 1]
    h = [1, 4, 6, 4, 1]
    y = np.convolve(x, h)
    print(y)

def part2_2():
    import numpy as np
    from sklearn.metrics import mean_squared_error as mse
    np.set_printoptions(precision=1, linewidth=300)
    from matplotlib import pyplot as plot

    # Q2.2 Show that linear convolution in time = point-by-point multiplication in frequency (FFT)
    # x = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 1]
    x = [1, 2, 3, 4, 5, 6, 5, 4, 3]
    h = [1, 4, 6, 4, 1]
    x1 = np.pad(x, (0, 4), 'constant')  # pad x=5-1
    h1 = np.pad(h, (0, 8), 'constant')  # pad h=9-1

    # Direct convolution
    y = np.convolve(x, h)
    print('\nDirect Convolution(t): ', y)

    # Point-by-point multiplication: pad, fft, pbp, ifft
    x2 = np.fft.fft(x1)  # fft xp
    h2 = np.fft.fft(h1)  # fft hp
    y1 = np.multiply(x2, h2)  # point-by-point multiplication
    y2 = np.fft.fft(y)  # fft of direct convolution
    y3 = np.fft.ifft(y1)
    y4 = np.fft.ifft(y2)
    print('FFT(t) real: ', np.real(y3))
    print('Direct Convolution(w): ', y2)  #
    print('FFT(w): ', y1)

    diff = mse(abs(y2), abs(y1))
    print('MSE(w)= ', diff)
    diff2 = mse(abs(y3), abs(y4))
    print('MSE(t)= ', diff2)
    print('\nappendix')
    print(y4)
    print('FFT(t): ', y3)


    plot.stem(np.real(y3)+y)  # np.real(y3) or y
    plot.xlim([-1, 16])  #[-1, x] where x = n of np.fft.fft(xn, n)
    plot.ylim([-1, 100])
    # plot.stem(y)
    plot.show()

def part2_3():

    import numpy as np
    from numpy import random
    from sklearn.metrics import mean_squared_error as mse
    np.set_printoptions(precision=1, linewidth=300)
    import time

    a = 102500
    b = 1024
    x = random.rand(a)
    h = random.rand(b)
    tic = time.time()
    y = np.convolve(x, h)
    toc = time.time()
    print('Direct Convolution(t): ', y)
    print('Direct Convolution time (s): ', toc - tic)
    t = toc - tic

    tic = time.time()
    x1 = np.pad(x, (0, b - 1), 'constant')
    h1 = np.pad(h, (0, a - 1), 'constant')
    x01 = np.fft.fft(x1, a + b - 1)
    h01 = np.fft.fft(h1, a + b - 1)
    y01 = np.multiply(x01, h01)
    y02 = np.fft.ifft(y01)
    toc = time.time()

    print('FFT(t): ', y02)
    print('FFT time(s): ', toc - tic)
    print('FFT time saved(s): ', (t - (toc - tic)))
    diff = mse(y, np.real(y02))
    print('DC vs FFT: ', diff)
    if diff < 1:
        print("results similar")

def part2_4():

    # q2.4 overlap and add method
    # implement 2.3 convolution via section method.
    # speed up the computation using FFT
    import numpy as np
    from numpy import random
    from sklearn.metrics import mean_squared_error as mse
    np.set_printoptions(precision=2, linewidth=500)
    import time

    a = 102500
    b = 1024
    x = random.rand(a)
    h = random.rand(b)

    # direct convolution
    tic = time.time()
    y = np.convolve(x, h)
    toc = time.time()
    print('Direct Convolution(t): ', y)
    print('Direct Convolution time (s): ', toc - tic)
    t = toc - tic

    # convolution by section
    tic = time.time()
    xspl = np.split(x, 100)  # splits 102500 into 1025 x 100 arrays, this function is limited for arrays that can be nicely split into integers
    # print('split: ', xspl)
    H = np.fft.fft(h, 2048)
    # print("H =", H)
    X = np.fft.fft(xspl, 2048)  # 2048-point DFT
    # print('X= ', X)
    Yr = np.multiply(X, H)
    # print('Y= ', Yr)
    yr = np.fft.ifft(Yr)
    # print('y= ', yr)
    yf = []
    yf.extend(yr[0][:2048 - 1023])
    # print('len(yf)=', len(yf))
    # print('yf)=', yf)
    for i in range(0, 99):
        yf.extend(yr[i][1023:]+yr[i+1][:2048-1023])
        # print('yf[', i, ']=', yf)
    yf.extend(yr[99][1025:])
    toc = time.time()
    # print('FFT section (t): ', yf)
    print('convolution by section time(s): ', toc - tic)
    # print('convolution by section(t): ', yf)
    print('time saved by FFT(s): ', t - toc + tic)
    diff = mse(y, np.abs(yf))
    print('mse(DC, FFT): ', diff)

    if diff < 10:
        print("results similar")

def part2_4clean():

    # q2.4 overlap and add method vs direct convolution
    import numpy as np
    from numpy import random
    from sklearn.metrics import mean_squared_error as mse
    np.set_printoptions(precision=2, linewidth=500)
    import time

    a = 102500
    b = 1024
    x = random.rand(a)
    h = random.rand(b)

    # direct convolution
    tic = time.time()
    y = np.convolve(x, h)
    toc = time.time()
    print('Direct Convolution(t): ', y)
    print('Direct Convolution time (s): ', toc - tic)
    t = toc - tic

    # convolution by section
    tic = time.time()
    xspl = np.split(x, 100)
    H = np.fft.fft(h, 2048)
    X = np.fft.fft(xspl, 2048)
    Yr = np.multiply(X, H)
    yr = np.fft.ifft(Yr)
    yf = []
    yf.extend(yr[0][:2048 - 1023])
    for i in range(0, 99):
        yf.extend(yr[i][1023:]+yr[i+1][:2048-1023])
    yf.extend(yr[99][1025:])
    toc = time.time()
    diff = mse(y, np.abs(yf))
    print('convolution by section time(s): ', toc - tic)
    # print('convolution by section(t): ', yf)
    print('time saved by FFT(s): ', t - toc + tic)
    print('mse(DC, FFT): ', diff)
    if diff < 10:
        print("results similar")

  # testing function
def npsplit():  # function testing
    import numpy as np
    x = np.arange(18)
    print(x)
    y = np.split(x, 3)
    print(y)
    y1 = np.split(x, [2])
    print(y1)
    y2 = np.stack(y)
    print(y2)
    print(np.multiply.reduce([2,3,5]))

  # assignment
def ass1_a3():
    import numpy as np
    from matplotlib import pyplot as plot
    # np.set_printoptions(precision=2, linewidth=100)

    # Q1.3
    x1 = [1, 8, 0, 7, 8, 6, 7, 3, 2, 16, 0, 14, 16, 12, 14, 6]
    y1 = np.fft.fft(x1, 16000)
    magn_x1 = abs(y1)
    sample_3khz = magn_x1[3000]
    print('|x|[3000] = ', sample_3khz)
    plot.stem(magn_x1)
    plot.xlim([-1, 16000])  #[-1, x] where x = n of np.fft.fft(xn, n)
    plot.ylim([-1, 130])
    plot.show()

if __name__ == '__main__':
    # part1()
    # part1_2()
    # part1_3()
    # part2_1()
    # part2_2()
    # part2_3()
    # part2_4()
    part2_4clean()
    # test()
    # npsplit()
    # ass1_a3()