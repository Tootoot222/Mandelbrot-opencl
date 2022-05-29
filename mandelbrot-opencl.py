from tracemalloc import start
from PIL import Image, ImageDraw
from matplotlib import pyplot
import numpy
import multiprocessing
import pyopencl
import os
import io
import pickle
import time
from mpmath import mpf, mpc, floor, linspace, log

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0'

colors = {}
it = 0
for c in range(100, 255, 2):
    colors[it] = (c, 0, 0)
    it += 1
for c in range(100, 255, 2):
    colors[it] = (0, c, 0)
    it += 1
for c in range(100, 255, 2):
    colors[it] = (0, 0, c)
    it += 1

for c in range(100, 255, 2):
    colors[it] = (c, c, 0)
    it += 1
for c in range(100, 255, 2):
    colors[it] = (0, c, c)
    it += 1
for c in range(100, 255, 2):
    colors[it] = (c, 0, c)
    it += 1
maxColors = it

def colorIterations(iterations):
    return colors[iterations % maxColors]
    #return (int(floor(255 * (iterations / maxIterations))), 0, 0)

def createImageBuffer(imageSize):
    imageBuffer = []
    for x in range(0, imageSize[0]):
        yBuffer = []
        for y in range(0, imageSize[1]):
            yBuffer.append((0, 0, 0))
        imageBuffer.append(yBuffer)
    return imageBuffer

def isPointInSet(x, y, maxIterations):
    point = mpc(x, y)
    z = mpc(x, y)
    for i in range(0, maxIterations):
        z = (z ** 2) + point
        #print(x, y, z, abs(z.real))
        if (abs(z.real) >= 2):
            #print(i, "iterations")
            return (False, i)
    return (True, maxIterations)

def isPointInSetArray(xArray, yArray, maxIterations):
    iterationsesArray = numpy.zeros(len(xArray), dtype=numpy.int32)
    xArray = xArray.astype(numpy.float32)
    yArray = yArray.astype(numpy.float32)

    ctx = pyopencl.create_some_context()
    queue = pyopencl.CommandQueue(ctx)

    mf = pyopencl.mem_flags
    realsCL = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xArray)
    imagsCL = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yArray)
    iterationsesCL = pyopencl.Buffer(ctx, mf.WRITE_ONLY, iterationsesArray.nbytes)

    prg = pyopencl.Program(ctx, """
        #include <pyopencl-complex.h>
        __kernel void multiply(
            ushort maxIterations,
            __global float *reals,
            __global float *imags,
            __global int *iterationses
        ) {
            int gid = get_global_id(0);
        
            cfloat_t point;
            point.real = reals[gid];
            point.imag = imags[gid];
            cfloat_t z;
            z.real = reals[gid];
            z.imag = imags[gid];
            int i;
            for (i = 0; i < maxIterations; i++) {
                z = cfloat_add(cfloat_mul(z, z), point);
                if (cfloat_abs(z) >= 2) {
                    iterationses[gid] = i + 1;
                    return;
                }
            }
            iterationses[gid] = 0;
        }
    """).build()

    prg.multiply(queue, iterationsesArray.shape, (32, 32),
                numpy.uint16(maxIterations),
                realsCL, imagsCL, iterationsesCL)

    iterationses = numpy.empty_like(iterationsesArray)
    pyopencl.enqueue_copy(queue, iterationses, iterationsesCL)
    return iterationses

def isPointInSetGenerated(worldDimensionsX, worldDimensionsY, imageSize, maxIterations, workGroupSize):
    iterationsesArray = numpy.zeros(imageSize, dtype=numpy.int32)

    ctx = pyopencl.create_some_context()
    queue = pyopencl.CommandQueue(ctx)

    mf = pyopencl.mem_flags
    iterationsesCL = pyopencl.Buffer(ctx, mf.WRITE_ONLY, iterationsesArray.nbytes)

    prg = pyopencl.Program(ctx, """
        #define PYOPENCL_DEFINE_CDOUBLE 1
        #include <pyopencl-complex.h>
        __kernel void multiply(
            double worldDimensionsX,
            double worldDimensionsY,
            double stepSizeX,
            double stepSizeY,
            ushort imageSizeX,
            ushort imageSizeY,
            ushort maxIterations,
            __global int *iterationses
        ) {
            int xIdx = get_local_id(0)+get_group_id(0)*get_local_size(0);
            int yIdx = get_local_id(1)+get_group_id(1)*get_local_size(1);
            int gid = yIdx + imageSizeY*xIdx;
        
            cdouble_t point;
            point.real = worldDimensionsX + (xIdx * stepSizeX);
            point.imag = worldDimensionsY + (yIdx * stepSizeY);
            cdouble_t z;
            z.real = point.real;
            z.imag = point.imag;
            int i;
            iterationses[gid] = 0;
            for (i = 0; i < maxIterations; i++) {
                z = cdouble_add(cdouble_mul(z, z), point);
                if (cdouble_abs(z) >= 2) {
                    iterationses[gid] = i + 1;
                    break;
                }
            }
        }
    """).build()

    prg.multiply(queue, iterationsesArray.shape, workGroupSize,
                numpy.double(worldDimensionsX[0]),
                numpy.double(worldDimensionsY[0]),
                numpy.double((worldDimensionsX[1] - worldDimensionsX[0]) / imageSize[0]),
                numpy.double((worldDimensionsY[1] - worldDimensionsY[0]) / imageSize[1]),
                numpy.uint16(imageSize[0]),
                numpy.uint16(imageSize[1]),
                numpy.uint16(maxIterations),
                iterationsesCL)

    iterationses = numpy.empty_like(iterationsesArray)
    pyopencl.enqueue_copy(queue, iterationses, iterationsesCL)
    return iterationses

def isPointInSetMatrix(xMatrix, yMatrix, maxIterations):
    iterationsesArray = numpy.zeros(xMatrix.shape, dtype=numpy.int32)
    xMatrix = xMatrix.astype(numpy.double)
    yMatrix = yMatrix.astype(numpy.double)

    ctx = pyopencl.create_some_context()
    queue = pyopencl.CommandQueue(ctx)

    mf = pyopencl.mem_flags
    realsCL = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xMatrix)
    imagsCL = pyopencl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yMatrix)
    iterationsesCL = pyopencl.Buffer(ctx, mf.WRITE_ONLY, iterationsesArray.nbytes)

    prg = pyopencl.Program(ctx, """
        #define PYOPENCL_DEFINE_CDOUBLE 1
        #include <pyopencl-complex.h>
        __kernel void multiply(
            ushort ySize,
            ushort maxIterations,
            __global double *reals,
            __global double *imags,
            __global int *iterationses
        ) {
            //int gid = get_global_id(0);
            int xIdx = get_local_id(0)+get_group_id(0)*get_local_size(0);
            int yIdx = get_local_id(1)+get_group_id(1)*get_local_size(1);
            int gid = yIdx + ySize*xIdx;
        
            cdouble_t point;
            point.real = reals[gid];
            point.imag = imags[gid];
            cdouble_t z;
            z.real = reals[gid];
            z.imag = imags[gid];
            int i;
            for (i = 0; i < maxIterations; i++) {
                z = cdouble_add(cdouble_mul(z, z), point);
                if (cdouble_abs(z) >= 2) {
                    iterationses[gid] = i + 1;
                    return;
                }
            }
            iterationses[gid] = 0;
        }
    """).build()

    prg.multiply(queue, iterationsesArray.shape, (1, 1),
                numpy.uint16(iterationsesArray.shape[1]), numpy.uint16(maxIterations),
                realsCL, imagsCL, iterationsesCL)

    iterationses = numpy.empty_like(iterationsesArray)
    pyopencl.enqueue_copy(queue, iterationses, iterationsesCL)
    return iterationses


def renderSetMatrix(imageNumber, imageBuffer, imageSize, maxIterations, worldDimensionsX, worldDimensionsY, workGroupSize):
    startTime = time.process_time()
    pointCount = 0
    pointTotalCount = 0
    iterationTotalCount = 0
    imageX = 0
    #print('assembling the matrix')
    #matrixX = [[x for y in linspace(worldDimensionsY[0], worldDimensionsY[1], imageSize[1], endpoint=False)] for x in linspace(worldDimensionsX[0], worldDimensionsX[1], imageSize[0], endpoint=False)]
    #matrixY = [[y for y in linspace(worldDimensionsY[0], worldDimensionsY[1], imageSize[1], endpoint=False)] for x in linspace(worldDimensionsX[0], worldDimensionsX[1], imageSize[0], endpoint=False)]
#    for x in linspace(worldDimensionsX[0], worldDimensionsX[1], imageSize[0], endpoint=False):
#        xCol = []
#        yCol = []
#        for y in linspace(worldDimensionsY[0], worldDimensionsY[1], imageSize[1], endpoint=False):
#            xCol.append(x)
#            yCol.append(y)
#        matrixX.append(xCol)
#        matrixY.append(yCol)
        
    #xMatrix = numpy.array(matrixX, dtype=numpy.double)
    #yMatrix = numpy.array(matrixY, dtype=numpy.double)
    #endTime = time.process_time()
    #runTime = endTime - startTime
    #print('matrix assembled in ' + str(runTime) + 's')

    startTime = time.process_time()
    print('sending the matrix')
    #iterationses = isPointInSetMatrix(xMatrix, yMatrix, maxIterations)
    iterationses = isPointInSetGenerated(worldDimensionsX, worldDimensionsY, imageSize, maxIterations, workGroupSize)
    print('matrix received')
    for imageX, iterationsX in enumerate(iterationses):
        for imageY, iterations in enumerate(iterationsX):
            pointCount += 1
            pointTotalCount += 1
            iterationCountFixed = numpy.int64((iterations - 1) % maxIterations)
            iterationTotalCount += iterationCountFixed
            if (iterations != 0): # 0 means that the maxIterations was reached
                imageBuffer[imageX][imageY] = colorIterations(iterations)
    print('matrix processed')
    endTime = time.process_time()
    runTime = endTime - startTime
    rate = 'Inf'
    if (runTime > 0):
        rate = iterationTotalCount / runTime

    print(str(imageNumber) + ': FINAL ' + str(iterationTotalCount) + ' iterations in ' + str(pointTotalCount) + ' rounds in ' + str(runTime)
        + ' is ' + str(rate))

    return iterationTotalCount

def renderSet(imageNumber, imageBuffer, imageSize, maxIterations, worldDimensionsX, worldDimensionsY):
    startTime = time.process_time()
    chunkStartTime = time.process_time()
    pointCount = 0
    pointTotalCount = 0
    iterationCount = 0
    iterationTotalCount = 0
    imageX = 0
    chunkPrintSize=4000000 * 12
    for x in linspace(worldDimensionsX[0], worldDimensionsX[1], imageSize[0], endpoint=False):
        imageY = 0
        imagArray = numpy.array([y for y in linspace(worldDimensionsY[0], worldDimensionsY[1], imageSize[1], endpoint=False)], dtype=numpy.float32)
        realArray = numpy.full(len(imagArray), x).astype(numpy.float32)
        iterationses = isPointInSetArray(realArray, imagArray, maxIterations)

        for iterations in iterationses:
#        for y in linspace(worldDimensionsY[0], worldDimensionsY[1], imageSize[1], endpoint=False):
#            #print(x, "", y)
#
#            pointIsInSet = isPointInSet(x, y, maxIterations)
            pointCount += 1
            pointTotalCount += 1
            iterationCountFixed = (iterations - 1) % maxIterations
            iterationCount += iterationCountFixed
            iterationTotalCount += iterationCountFixed
#            if (not pointIsInSet[0]):
#                imageBuffer[imageX][imageY] = colorIterations(pointIsInSet[1])
            if (iterations != 0): # 0 means that the maxIterations was reached
                imageBuffer[imageX][imageY] = colorIterations(iterations)
            imageY += 1
        imageX += 1
        if (iterationCount > int(chunkPrintSize)):
            chunkEndTime = time.process_time()
            chunkRunTime = chunkEndTime - chunkStartTime
            print(str(imageNumber) + ': ' + str(iterationCount) + ' iterations in '
                + str(pointCount) + '/' + str(pointTotalCount) + '/' + str(imageSize[0] * imageSize[1])
                + ' rounds in ' + str(chunkRunTime)
                + ' is ' + str(iterationCount / chunkRunTime)
                + '/' + str(iterationTotalCount / (chunkEndTime - startTime))
            )
            pointCount = 0
            iterationCount = 0
            chunkStartTime = time.process_time()
    endTime = time.process_time()
    runTime = endTime - startTime
    rate = 'Inf'
    if (runTime > 0):
        rate = iterationTotalCount / runTime

    print(str(imageNumber) + ': FINAL ' + str(iterationTotalCount) + ' iterations in ' + str(pointTotalCount) + ' rounds in ' + str(runTime)
        + ' is ' + str(rate))

    return iterationTotalCount

def renderNewSet(imageNumber, imageResults, imageSize, maxIterations, worldDimensionsX, worldDimensionsY, workGroupSize):
    start = time.process_time()
    print(str(imageNumber) + ': render thread', imageNumber, 'processing image of size', imageSize, 'with workers of size', workGroupSize, 'of coordinates from', worldDimensionsX, 'to', worldDimensionsY, 'with max iterations', maxIterations)
    imageBuffer = createImageBuffer(imageSize)
    iterationCount = renderSetMatrix(imageNumber, imageBuffer, imageSize, maxIterations, worldDimensionsX, worldDimensionsY, workGroupSize)
    #imageResults.put((imageNumber, imageBuffer))
    imageResults[imageNumber] = (imageBuffer, iterationCount)
    end = time.process_time()
    elapsed = end - start
    rate = 'Inf'
    if (elapsed > 0):
        rate = iterationCount / elapsed
    print(str(imageNumber) + ': render thread ' + str(imageNumber) + 'completed ' + str(iterationCount) + ' iterations in ' + str(elapsed) + ' ' + str(rate))
    return imageBuffer
    #with io.open('set_' + str(imageNumber), 'wb') as imageFile:
    #    pickle.dump(imageBuffer, imageFile)
    #pyplot.figure(figsize = imageSize, dpi = imageZoom)
    #pyplot.imshow(imageBuffer)
    #pyplot.savefig('fig.png')
    #pyplot.axis('off')
    #pyplot.imshow(imageBuffer)
    #pyplot.axis('off')
    #pyplot.imsave('fig.png', imageBuffer)
    #return imageBuffer

def renderParallelSet(imageSize, threadSize, maxIterations, worldDimensionsX, worldDimensionsY, workGroupSize):
    startTime = time.time()
    chunkSize = (int(imageSize / threadSize[0]), int(imageSize / threadSize[1]))
    threads = []

    imageResults = multiprocessing.Manager().dict()
    imageNumber = 0
    imageX = 0
    for x in linspace(worldDimensionsX[0], worldDimensionsX[1], threadSize[0], endpoint=False):
        imageY = 0
        for y in linspace(worldDimensionsY[0], worldDimensionsY[1], threadSize[1], endpoint=False):
            # performance test: is it faster to get the pickled return value, or just pass it the blank image buffer as a shared object?
            #imageBuffer = createImageBuffer((chunkRowSize, chunkColSize))
            #print(imageX, imageY)
            #imageBuffers[imageX][imageY] = imageBuffer
            thread = multiprocessing.Process(
                target = renderNewSet,
                args = (
                    imageNumber,
                    imageResults,
                    chunkSize,
                    maxIterations,
                    (x, x + ((worldDimensionsX[1] - worldDimensionsX[0]) / threadSize[0])),
                    (y, y + ((worldDimensionsY[1] - worldDimensionsY[0]) / threadSize[1])),
                    workGroupSize
                )
            )
            threads.append(thread)
            thread.start()
            imageY += 1
            imageNumber += 1
        imageX += 1


    print("join start")
    for t in threads:
        t.join()
    print("join done")
#    for r in range(0, len(results)):
#        print(r, results[r])
#        imageBuffers[r % threadSize[0]][int(floor(r / threadSize[1]))] = results[r].get()

    #imageResultsMap = {imageResult[0]: imageResult[1] for imageResult in imageResults}
#    imageResultsMap = {}
#    while not imageResults.empty():
#        imageResult = imageResults.get()
#        imageResultsMap[imageResult[0]] = imageResult[1]
    totalIterations = 0
    imageBuffers = [] #createImageBuffer((imageSize, imageSize))
    for x in range(0, threadSize[0]):
        yBuffer = []
        for y in range(0, threadSize[1]):
            resultIndex = (x * threadSize[1]) + y
            yBuffer.append(imageResults[resultIndex][0])
            totalIterations += imageResults[resultIndex][1]
            #with io.open('set_' + str(), 'rb') as imageFile:
            #    yBuffer.append(pickle.load(imageFile))
        imageBuffers.append(yBuffer)
    endTime = time.time()
    print('collected results from ' + str(totalIterations) + ' iterations in ' + str(endTime - startTime) + 's: '
        + str(totalIterations / (endTime - startTime)) + ' avg ' + str((totalIterations / (endTime - startTime)) / (threadSize[0] * threadSize[1])))

    imageBuffer = []
    for y in range(0, threadSize[1]):
        resultX = imageBuffers[0][y]
        for x in range(1, threadSize[1]):
            resultX = numpy.concatenate((resultX, imageBuffers[x][y]), axis=0)
        if (y == 0):
            imageBuffer = resultX
        else:
            imageBuffer = numpy.concatenate((imageBuffer, resultX), axis=1)
    return imageBuffer

    #data = write_png(imageBuffer, imageSize, imageSize)
    #with open("my_image.png", 'wb') as fh:
    #    fh.write(data)

    #pyplot.figure(figsize = (imageSize, imageSize), dpi = imageZoom)
    #pyplot.imshow(numpy.rot90(imageBuffer))
    #pyplot.imshow(imageBuffer)
    #pyplot.axis('off')
    #pyplot.savefig('fig.png')


if __name__ == '__main__':
    imageSize = 1200
    threadSize = (2, 2)
    workGroupImageSize = 75
    workGroupSize = (int((imageSize / threadSize[0]) / workGroupImageSize), int((imageSize / threadSize[1]) / workGroupImageSize))

    #imageZoom = int(floor(3000 / imageSize))

    #centerCoordinate = (mpf('0'), mpf('0'))
    #worldSize = mpf('2') #radius
    #centerCoordinate = (mpf('0'), mpf('0'))
    #worldSize = mpf('1') #radius
    #centerCoordinate = (mpf('0'), mpf('0'))
    #worldSize = mpf('1') #radius
    #centerCoordinate = (mpf('-1.39600724'), mpf('-0.0066233'))
    #worldSize = mpf('0.00001') #radius
    #centerCoordinate = (mpf('-0.909066765'), mpf('0.2656176'))
    #worldSize = mpf('0.00001') #radius

    #target
    #centerCoordinate = (mpf('-0.7570126264005'), mpf('0.0620842044993'))
    #worldSize = mpf('0.0000000000001') #radius
    #maxIterations = 4500

    #GPU Zoom 1
    #centerCoordinate = (mpf('-0.7570126264005'), mpf('0.0620842044993'))
    #Gpu Zoom 2
    #centerCoordinate = (mpf('-1.9414546650019524'), mpf('-0.0057152970004928508'))
    centerCoordinate = (mpf('0.253270382'), mpf('-0.00029617488476434698'))

    #worldSize = mpf('0.0000000000001') #radius
    #worldSizeTarget = mpf('1') #radius
    #worldSizeStart = mpf('2') #radius
    #worldSizeTarget = mpf('0.00000000000001') #radius
    worldSizeTarget = mpf('0.000000000000001') #radius
    #worldSizeStart = mpf('4.4721359549995798e-7') #radius
    worldSizeStart = mpf('2') #radius
    #worldSizeStart = mpf('0.0033873321941200185')
    #worldSizeTarget = mpf('0.01') #radius
    #worldSizeStart = mpf('0.01') #radius
    maxIterations = 8000
    minIterations = 40
    crashIterations = 0#maxIterations - 100
    totalIterations = maxIterations - minIterations
    onlyEvery = 0

    skipIterations = (totalIterations - (crashIterations - minIterations)) / 5
    worldSizes = []

    iterationCount = minIterations
    iterations = minIterations
    for worldSize in map(lambda x: 10 ** x, linspace(log(worldSizeStart, 10), log(worldSizeTarget, 10), totalIterations)):
        worldDimensionsX = (centerCoordinate[0] - worldSize, centerCoordinate[0] + worldSize)
        worldDimensionsY = (centerCoordinate[1] - worldSize, centerCoordinate[1] + worldSize)

        if (iterations < crashIterations):
            iterations += 1
            iterationCount += 1
            if (iterationCount > skipIterations):
                print('renderParallelSet(', imageSize, threadSize, iterations, worldDimensionsX, worldDimensionsY, workGroupSize)
                print('skipping, to resume at crash iterations ' + str(crashIterations))
                iterationCount = 0
            continue
        if (onlyEvery == 0 or (iterations % onlyEvery == 0)):
            print('renderParallelSet(', imageSize, threadSize, iterations, worldDimensionsX, worldDimensionsY, workGroupSize)
            imageBuffer = renderParallelSet(imageSize, threadSize, iterations, worldDimensionsX, worldDimensionsY, workGroupSize)

            filename = 'C:/Users/tybrown/Projects/Mandelbrot-opencl/output' + str(iterations) + '.png'
            print("writing to", filename)
            Image.fromarray(numpy.rot90(numpy.array(imageBuffer, dtype=numpy.uint8))).save(filename)
        iterations += 1
        iterationCount += 1