import numpy as np
import matplotlib.pyplot as plt

A1 = np.array([[1000,1],[0,1],[0,0]])
A2 = np.array([[1,0,0],[1,0,0],[0,0,1]])


U1,S1,Vt1 = np.linalg.svd(A1, full_matrices = False)
U2,S2,Vt2 = np.linalg.svd(A2, full_matrices = False)

B = np.array([[2,0,0], [1,0,1], [0,1,0]])

def truncSVD(U,S,Vt,d):
    assert d != 0, "d kan ikke være lik 0"
    
    if d > U.shape[1]:
        
        W = U
        Sd = S
        Vdt = Vt[:U.shape[1]]
        H = np.diag(Sd).dot(Vdt)

    else:
        W = U[:, :d]
        Sd = S[:d]
        Vdt = Vt[:d]
        H = np.diag(Sd).dot(Vdt)


    return W, H

def orthproj(W, B):
    W_t = np.transpose(W)
    antallRader, antallKolonner = B.shape
    projection = np.empty((antallRader, antallKolonner))
    for i in range(antallKolonner):
        B_i = B[::, i]
        first = np.dot(W_t, B_i)
        projectionCol = np.dot(W, first)
        projection[::,i] = projectionCol
    
    return projection

proj = orthproj(U1, B)
print(proj)

proj2 = orthproj(U2, B)
print(proj2)

def distanceProj(B, projection):
    antallRader, antallKolonner = B.shape
    distanceProjection = np.empty((1, antallKolonner))
    for i in range(antallKolonner):
        dist = np.linalg.norm(B[::,i] - projection[::, i])
        distanceProjection[::,i] = dist
    
    return distanceProjection

distance = distanceProj(B, proj)
print(distance)

delta = 10**-10

def nnproj(W, A):
    maxiter = 50
    global delta
    wRow, wCol = W.shape
    aRow, aCol = A.shape
    matrise1 = np.dot(np.transpose(W), A)
    matrise2 = np.dot(np.transpose(W), W)
    H_k = np.random.uniform(0,1, (wCol, aCol))
    
    for i in range(maxiter):
        H_k = H_k * matrise1 / (np.dot(matrise2, H_k) + delta)
    
    hRow, hCol = H_k.shape
    projectionD = np.empty((aRow,hCol))
    
    for i in range(hCol):
        projMat = np.dot(W, H_k[::,i])
        projectionD[::,i] = projMat
    
    distance = distanceProj(A, projectionD)

    return H_k, projectionD, distance

height, projection, dist = nnproj(A1, B)
height2, projection2, dist2 = nnproj(A2, B)
print(height)
print(projection)
print(dist) #Første element til distansematrisen blir mer nøyaktig jo flere iterasjoner, altså nærmere null.

#Oppgave 2

# Load the data and resclae
train = np.load('train.npy')/255.0
test = np.load('test.npy')/255.0

# Shapes are (number of pixels, number of classes, number of data)
print(train.shape) # Should be (784,10,5000)
print(test.shape) # Should be (784,10,800)

def plotimgs(imgs, nplot = 4):
    """
    Plots the nplot*nplot first images in imgs on an nplot x nplot grid. 
    Assumes heigth = width, and that the images are stored columnwise
    input:
        imgs: (height*width,N) array containing images, where N > nplot**2
        nplot: integer, nplot**2 images will be plotted
    """

    n = imgs.shape[1]
    m = int(np.sqrt(imgs.shape[0]))

    assert(n > nplot**2), "Need amount of data in matrix N > nplot**2"

    # Initialize subplots
    fig, axes = plt.subplots(nplot,nplot)

    # Set background color
    plt.gcf().set_facecolor("lightgray")

    # Iterate over images
    for idx in range(nplot**2):

        # Break if we go out of bounds of the array
        if idx >= n:
            break

        # Indices
        i = idx//nplot; j = idx%nplot

        # Remove axis
        axes[i,j].axis('off')

        axes[i,j].imshow(imgs[:,idx].reshape((m,m)), cmap = "gray")
    
    # Plot

    fig.tight_layout()
    plt.show()

# Plot first 16 images of the zero integer
plotimgs(train[:,0,:], nplot = 4)

# Plot the second image of the 2 digit
# Note that we have to reshape it to be 28 times 28!
plt.imshow(train[:, 2, 1].reshape((28,28)), cmap = 'gray')
plt.axis('off')
plt.show()

n = 1000 # Number of datapoints
c = 4 # Class

A = train[:,c,:n]

print(A.shape) # Expect (784,n)

#Oppgave 2b

d = 16
uBilde, sBilde, vtBilde = np.linalg.svd(A)
udBilde, hdBilde = truncSVD(uBilde, sBilde, vtBilde, d)
A_new = udBilde.dot(hdBilde)
plotimgs(udBilde, nplot = 2)
plt.title('Singular values from highest to lowest')
plt.semilogy(sBilde)
plt.show()

#Oppgave 2c

dArray = np.array([16, 32, 64, 128])

for i in dArray:
    dValue = i
    udBildeLoop, hdBildeLoop = truncSVD(uBilde, sBilde, vtBilde, i)
    projBilde = orthproj(udBildeLoop, A)
    plotimgs(projBilde, nplot = 2)
    plotimgs(A, nplot = 2)
    plt.show()

#Oppgave 2d

dValueArray = np.linspace(1,784, 28, dtype=int)
resultArray = np.empty(len(dValueArray))
for i in range(len(dValueArray)):
    new_Ud, new_Hd = truncSVD(uBilde, sBilde, vtBilde, dValueArray[i])
    newProjBilde = orthproj(new_Ud, A)
    diff = A - newProjBilde
    resultVal = (np.linalg.norm(diff, 'fro')) ** 2
    resultArray[i] = resultVal

plt.semilogy(resultArray)
plt.title(f'Frobenius norm squared with varying d values')
plt.show()

#Oppgave 2e

d_e = 32

wPlusIndex = np.random.choice(A.shape[1],d_e,replace=False)
nnA = np.empty((A.shape[0], A.shape[1]))

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        nnA[i,j] = abs(A[i,j])

wPlus = np.empty((784, d_e))

for i in range(d_e):
    wPlus[::,i] = nnA[:, wPlusIndex[i]]

H_ke, projE, distE = nnproj(wPlus, nnA)
projE_SVD = orthproj(wPlus,A)

plotimgs(projE, nplot = 4)
print('Sammenlignet med SVD: ')
plotimgs(projE_SVD, nplot =4)

#Oppgave 2f

grid = np.logspace(1,3, 28, dtype = int)
resultGrid = np.empty(len(grid))

for i in range(len(grid)):
    plusUd, plusHd = truncSVD(uBilde, sBilde, vtBilde, grid[i])
    H_k_plus, plusProj, distPlus = nnproj(plusUd, nnA)
    diffPlus = A - plusProj
    resultGridVal = (np.linalg.norm(diffPlus, 'fro')) ** 2
    resultGrid[i] = resultGridVal

plt.semilogy(resultGrid, label='EMNF')
plt.semilogy(resultArray, label='SVD')
plt.legend()
plt.title(f'Frobenius norm squared with varying d values')
plt.show()

def generate_test(test, digits = [0,1,2], N = 800):
    """
    Randomly generates test set.
    input:
        test: numpy array. Should be the test data loaded from file
        digits: python list. Contains desired integers
        N: int. Amount of test data for each class
    output:
        test_sub: (784,len(digits)*N) numpy array. Contains len(digits)*N images
        test_labels: (len(digits)*N) numpy array. Contains labels corresponding to the images of test_sub
    """

    assert N <= test.shape[2] , "N needs to be smaller than or equal to the total amount of available test data for each class"

    assert len(digits)<= 10, "List of digits can only contain up to 10 digits"

    # Arrays to store test set and labels
    test_sub = np.zeros((test.shape[0], len(digits)*N))
    test_labels = np.zeros(len(digits)*N)

    # Iterate over all digit classes and store test data and labels
    for i, digit in enumerate(digits):
        test_sub[:, i*N:(i+1)*N] = test[:,digit,:]
        test_labels[i*N:(i+1)*N] = digit

    # Indexes to be shuffled 
    ids = np.arange(0,len(digits)*N)

    # Shuffle indexes
    np.random.shuffle(ids)

    # Return shuffled data 
    return test_sub[:,ids], test_labels[ids]

digits = [0,1,2]

A_test, A_labels = generate_test(test, digits = digits, N = 800)
print("Test data shape: ", A_test.shape) # Should be (784,2400)
print("Test labels shape: ", A_labels.shape) # Should be (2400)
print("First 16 labels: ", A_labels[:16])
plotimgs(A_test, nplot = 4)

#Oppgave 3a

def testingSVD(B, dicList):
    distDicList = np.empty( (len(dicList), B.shape[1]))
    projDicList = np.empty((len(dicList), B.shape[0], B.shape[1]))
    predLabel = np.empty(B.shape[1])
    for i in range(len(dicList)):
        projDic = orthproj(dicList[i], B)
        projDicList[i] = projDic
        distDic = distanceProj(B, projDic)
        distDicList[i] = distDic
    
    for j in range(B.shape[1]):
        index = np.argmin(distDicList[:,j])
        predLabel[j] = index
    
    return projDicList, distDicList, predLabel

def testingENMF(B, dicList):
    distDicList = np.empty((len(dicList), B.shape[1]))
    projDicList = np.empty((len(dicList), B.shape[0], B.shape[1]))
    predLabel = np.empty(B.shape[1])
    for i in range(len(dicList)):
        H_Dic, projDic, distDic = nnproj(dicList[i], B)
        projDicList[i] = projDic
        distDicList[i] = distDic
    for j in range(B.shape[1]):
        index = np.argmin(distDicList[:,j])
        predLabel[j] = index
    
    return projDicList, distDicList, predLabel

def Accuracy(labels, predictionLabels):
    right = 0
    for i in range(len(labels)):
        if labels[i] == predictionLabels[i]:
            right += 1
        else:
            continue
    accuracy = right/len(labels)
    return accuracy

def Recall(labels, predictionLabels, digits):
    listeLabels = []
    amountRight = np.zeros(len(digits))
    for i in range(len(labels)):
        listeLabels.append(labels[i])
        if labels[i] == predictionLabels[i]:
            for j in digits:
                if labels[i] == j:
                    sumNum = amountRight[j]
                    sumNum += 1
                    amountRight[j] = sumNum 
    
    for k in range(len(amountRight)):
        print(f'Recall {digits[k]} = ', amountRight[k]/listeLabels.count(digits[k]))

#Oppgave 3b

zero = train[:, 0, :2400]
one = train[:, 1, :2400]
two = train[:, 2, :2400]

nnZero = np.empty((zero.shape[0], zero.shape[1]))
nnOne = np.empty((one.shape[0], one.shape[1]))
nnTwo = np.empty((two.shape[0], two.shape[1]))

for i in range(zero.shape[0]):
    for j in range(zero.shape[1]):
        nnZero[i,j] = abs(zero[i,j])
        nnOne[i,j] = abs(one[i,j])
        nnTwo[i,j] = abs(two[i,j])

zeroU, zeroS, zeroVt = np.linalg.svd(zero)
oneU, oneS, oneVt = np.linalg.svd(one)
twoU, twoS, twoVt = np.linalg.svd(two)

zeroUd, zeroH = truncSVD(zeroU, zeroS, zeroVt, 32)
oneUd, oneH = truncSVD(oneU, oneS, oneVt, 32)
twoUd, twoH = truncSVD(twoU, twoS, twoVt, 32)

nnZeroU, nnZeroS, nnZeroVt = np.linalg.svd(nnZero)
nnOneU, nnOneS, nnOneVt = np.linalg.svd(nnOne)
nnTwoU, nnTwoS, nnTwoVt = np.linalg.svd(nnTwo)

nnZeroUd, nnZeroH = truncSVD(nnZeroU, nnZeroS, nnZeroVt, 32)
nnOneUd, nnOneH = truncSVD(nnOneU, nnOneS, nnOneVt, 32)
nnTwoUd, nnTwoH = truncSVD(nnTwoU, nnTwoS, nnTwoVt, 32)

dictionaryList = np.array([zeroUd, oneUd, twoUd])
nnDictionaryList = np.array([nnZeroUd, nnOneUd, nnTwoUd])

projDictionaryList, distanceList, predictionLabel = testingSVD(A_test,dictionaryList)
print(predictionLabel[:16])

nnProjDictionaryList, nnDistanceList, nnPredictionLabel = testingENMF(A_test, nnDictionaryList)
print(nnPredictionLabel[:16])

acc = Accuracy(A_labels, predictionLabel)
print("Accuracy (SVD): ", acc)

nnAcc = Accuracy(A_labels, nnPredictionLabel)
print("Accuracy (ENMF): ", nnAcc)

Recall(A_labels, predictionLabel, digits)

Recall(A_labels, nnPredictionLabel, digits)

#Oppgave 3c

def chosenInteger(integer):

    minValue = np.min(distanceList[integer])
    indeks = np.where(distanceList[integer] == minValue)

    nnMinValue = np.min(nnDistanceList[integer])
    nnIndeks = np.where(nnDistanceList[integer] == nnMinValue)


    print('\nPlot SVD: ')

    plt.imshow(A_test[:,indeks].reshape((28,28)), cmap = 'gray')
    plt.axis('off')
    plt.show()

    print('Plot ENMF: ')

    plt.imshow(A_test[:, nnIndeks].reshape((28,28)), cmap = 'gray')
    plt.axis('off')
    plt.show()

    print('Plot SVD projection: ')

    plt.imshow(projDictionaryList[integer, :, indeks].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    plt.show()

    print('Plot ENMF projection: ')

    plt.imshow(nnProjDictionaryList[integer, :, nnIndeks].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    plt.show()

chosenInteger(1)

#Oppgave 3d

def findMisclassified(labels, predictionLabels, integer):
    for i in range(len(labels)):
        if (labels[i] != predictionLabels[i]) and (predictionLabels[i] == float(integer)):
            return i
        else:
            continue

misIndex = findMisclassified(A_labels, predictionLabel, 0)
plt.imshow(A_test[:, misIndex].reshape(28,28), cmap = 'gray')
plt.axis('off')
plt.show()

#Oppgave 3e

def expandDicLists():

    amountDigits = input('How many digits do you want to be classified? (Digit between 4 and 10)')
    digits = []
    for i in range(int(amountDigits)):
        digits.append(i)
    
    
    dictionaryList = np.empty((len(digits), 784, 32))
    nnDictionaryList = np.empty((len(digits), 784, 32))
    
    for j in range(len(digits)):
        numImage = train[:, j, : 800 * len(digits)]
        nnNumImage = np.abs(numImage)

        numImageU, numImageS, numImageVt = np.linalg.svd(numImage)
        nnNumImageU, nnNumImageS, nnNumImageVt = np.linalg.svd(nnNumImage)

        numImageUd, numImageH = truncSVD(numImageU, numImageS, numImageVt, 32)
        nnNumImageUd, nnNumImageH = truncSVD(nnNumImageU, nnNumImageS, nnNumImageVt, 32)

        dictionaryList[j] = numImageUd
        nnDictionaryList[j] = nnNumImageUd
    

    return digits, dictionaryList, nnDictionaryList

digits2, dictionaryList2, nnDictionaryList2 = expandDicLists()
A2_test, A2_labels = generate_test(test, digits = digits2, N = 800)

   
expProjDictionaryList, expDistanceList, expPredictionLabel = testingSVD(A2_test, dictionaryList2)
nnExpProjDictionaryList, nnExpDistanceList, nnExpPredictionLabel = testingENMF(A2_test, nnDictionaryList2)

expAcc = Accuracy(A2_labels, expPredictionLabel)
nnExpAcc = Accuracy(A2_labels, nnExpPredictionLabel)

print("Accuracy (SVD): ", expAcc)
print("Accuracy (ENMF): ", nnExpAcc)


print('\nSVD: ')
Recall(A2_labels, expPredictionLabel, digits2)

print('\nENMF')
Recall(A2_labels, nnExpPredictionLabel, digits2)

#Oppgave 3f

delta = 10**(-2)

def dFromAccuracy():

    dGridArray = np.zeros(10)

    for i in range(1, 11):
        dGridArray[i - 1] = 2 ** i

    accuracyArray = np.zeros(len(dGridArray))
    nnAccuracyArray = np.zeros(len(dGridArray))

    for i in range(len(dGridArray)):
        digitsArr, dictionaryListArr, nnDictionaryListArr = expandDicLists(3, int(dGridArray[i]))
        aArr_test, aArr_labels = generate_test(test, digits = digitsArr, N = 800)

        arrProjDictionaryList, arrDistanceList, arrPredictionLabel = testingSVD(aArr_test, dictionaryListArr)
        nnArrProjDictionaryList, nnArrDistanceList, nnArrPredictionLabel = testingENMF(aArr_test, nnDictionaryListArr)

        arrAcc = Accuracy(aArr_labels, arrPredictionLabel)
        nnArrAcc = Accuracy(aArr_labels, nnArrPredictionLabel)

        accuracyArray[i] = arrAcc
        nnAccuracyArray[i] = nnArrAcc

    plt.title('Accuracy from d values')
    plt.plot(dGridArray, accuracyArray)
    plt.xlabel('d')
    plt.ylabel('Accuracy')
    plt.plot(dGridArray, nnAccuracyArray)

dFromAccuracy()