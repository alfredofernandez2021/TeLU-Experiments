import statistics as stats
import numpy as np
from math import isnan


#print("hi")
Stability_Trials = []


#Read ResNet RMSprop Trials
for j in range(20,45,2):

  for i in range(0,10):
    tempFile = open('MLP_MNIST_Trial_'+str(j)+'_'+str(i)+'.csv', 'r')
    tempTrial = []
  #print(i)

  #Read lines
    TeLUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    MishTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    ReLUTrainingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

    TeLUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    MishValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    ReLUValidationAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

    TeLUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    MishTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    ReLUTrainingError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

    TeLUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    MishValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    ReLUValidationError = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

    TeLUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    MishTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    ReLUTrainingRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

    TeLUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    MishValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    ReLUValidationRuntime = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

    TeLUTestingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    MishTestingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]
    ReLUTestingAccuracy = [float(num) for num in tempFile.readline().replace("\"","").split(",")]

  #append CSV lines to temp trial list of metrics
    tempTrial.append(TeLUTrainingAccuracy)
    tempTrial.append(MishTrainingAccuracy)
    tempTrial.append(ReLUTrainingAccuracy)

    tempTrial.append(TeLUValidationAccuracy)
    tempTrial.append(MishValidationAccuracy)
    tempTrial.append(ReLUValidationAccuracy)

    tempTrial.append(TeLUTrainingError)
    tempTrial.append(MishTrainingError)
    tempTrial.append(ReLUTrainingError)

    tempTrial.append(TeLUValidationError)
    tempTrial.append(MishValidationError)
    tempTrial.append(ReLUValidationError)

    tempTrial.append(TeLUTrainingRuntime)
    tempTrial.append(MishTrainingRuntime)
    tempTrial.append(ReLUTrainingRuntime)

    tempTrial.append(TeLUValidationRuntime)
    tempTrial.append(MishValidationRuntime)
    tempTrial.append(ReLUValidationRuntime)

    tempTrial.append(TeLUTestingAccuracy)
    tempTrial.append(MishTestingAccuracy)
    tempTrial.append(ReLUTestingAccuracy)

  #append trial metric lists to trials lists
    Stability_Trials.append(tempTrial)
    tempFile.close()

#Prepare data from not-a-number values #REWORK??? todo: todo
# for i in range(0,len(Stability_Trials)):
#   for j in range(len(Stability_Trials[i][6])):
#     if isnan(Stability_Trials[i][4][j]):
#       Stability_Trials[i][4][j] = 1000.
#     if isnan(Stability_Trials[i][5][j]):
#       Stability_Trials[i][5][j] = 1000.

  print("MLP Stability:")
  print(j)

  TeLUTestingAccuracy = stats.mean([Stability_Trials[i][18][0] for i in range(0,10)])
  TeLUConverganceTestingStdv = stats.stdev([Stability_Trials[i][18][0] for i in range(0,10)])
  print("TeLU test acc: ", TeLUTestingAccuracy, " stdv: ", TeLUConverganceTestingStdv)

  MishTestingAccuracy = stats.mean([Stability_Trials[i][19][0] for i in range(0,10)])
  MishConverganceTestingStdv = stats.stdev([Stability_Trials[i][19][0] for i in range(0,10)])
  print("Mish test acc: ", MishTestingAccuracy, " stdv: ", MishConverganceTestingStdv)

  ReLUTestingAccuracy = stats.mean([Stability_Trials[i][20][0] for i in range(0,10)])
  ReLUConverganceTestingStdv = stats.stdev([Stability_Trials[i][20][0] for i in range(0,10)])
  print("ReLU test acc: ", ReLUTestingAccuracy, " stdv: ", ReLUConverganceTestingStdv)

  Stability_Trials.clear()
  print()

#Show TeLU metrics for RMSprop on ResNet:
# for i in range(5):
#   print(Tiny_ResNet_RMSprop_Trials[i][43])
# TeLUTestingAccuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][12][0] for i in range(5)])
# TeLUConverganceTestingStdv = stats.stdev([Tiny_ResNet_RMSprop_Trials[i][12][0] for i in range(5)])

# TeLUTesting5Accuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][12][1] for i in range(5)])
# TeLUConvergance5TestingStdv = stats.stdev([Tiny_ResNet_RMSprop_Trials[i][12][1] for i in range(5)])

# TeLUConverganceValidationAccuracy = stats.mean([np.max(Tiny_ResNet_RMSprop_Trials[i][2]) for i in range(5)])
# TeLUConverganceValidationMedian = stats.median([np.max(Tiny_ResNet_RMSprop_Trials[i][2]) for i in range(5)])
# TeLUConverganceValidationStdv = stats.stdev([np.max(Tiny_ResNet_RMSprop_Trials[i][2]) for i in range(5)])

# TeLUConclusiveValidationAccuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][2][-1] for i in range(5)])
# TeLUStabilityIndex = TeLUConclusiveValidationAccuracy / TeLUConverganceValidationAccuracy

# TeLUConverganceEpochIndeces = [np.argmax(Tiny_ResNet_RMSprop_Trials[i][2]) for i in range(5)]
# TeLUConverganceEpochFloats = [float(TeLUConverganceEpochIndex) for TeLUConverganceEpochIndex in TeLUConverganceEpochIndeces]
# TeLUConverganceEpochAverage = stats.mean(TeLUConverganceEpochFloats)

# TeLUConverganceTrainingAccuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][0][TeLUConverganceEpochIndeces[i]] for i in range(5)])

# TeLUFittingIndex = (100-TeLUConverganceTrainingAccuracy)/(100-TeLUTestingAccuracy)

# TeLUMinTrainingError = stats.mean([np.min(Tiny_ResNet_RMSprop_Trials[i][4]) for i in range(5)])

# TeLUMeanTrainingTime = stats.mean([stats.mean(Tiny_ResNet_RMSprop_Trials[i][8]) for i in range(5)])

# TeLUMeanValidationTime = stats.mean([stats.mean(Tiny_ResNet_RMSprop_Trials[i][10]) for i in range(5)])

# print(f"Average TeLU convergance Training accuracy: {TeLUConverganceTrainingAccuracy:.3f}")
# print(f"Average TeLU convergance Validation accuracy: {TeLUConverganceValidationAccuracy:.3f}")
# # print(f"Average TeLU convergance Testing Top-1 accuracy: {TeLUTestingAccuracy:.3f}")
# # print(f"Average TeLU convergance Testing Top-5 accuracy: {TeLUTesting5Accuracy:.3f}")
# print(f"Average TeLU conclusive Validation accuracy: {TeLUConclusiveValidationAccuracy:.3f}")
# #print(f"Median TeLU convergance Testing accuracy: {TeLUConverganceTestingMedian:.3f}")
# #print(f"Stdv of TeLU convergance Validation accuracy: {TeLUConverganceValidationStdv:.3f}")
# print(f"Average TeLU convergance epoch: {TeLUConverganceEpochAverage:.3f}")
# print(f"Average TeLU fitting index: {TeLUFittingIndex:.3f}")
# # print(f"Stdv of TeLU convergance test Top-1 accuracy: {TeLUConverganceTestingStdv:.3f}")
# # print(f"Stdv of TeLU convergance test Top-5 accuracy: {TeLUConvergance5TestingStdv:.3f}")
# #print(f"Average TeLU minimum training error: {TeLUMinTrainingError:.3f}")
# #print(f"Average TeLU epoch runtime: {TeLUMeanTrainingTime:.3f}")
# #print(f"Average TeLU testing runtime: {TeLUMeanTestingTime:.3f}")
# print()


#Show ReLU metrics for RMSprop on ResNet:
# for i in range(5):
#   print(Tiny_ResNet_RMSprop_Trials[i][43])
# ReLUTestingAccuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][13][0] for i in range(5)])
# ReLUConverganceTestingStdv = stats.stdev([Tiny_ResNet_RMSprop_Trials[i][13][0] for i in range(5)])

# ReLUTesting5Accuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][13][1] for i in range(5)])
# ReLUConvergance5TestingStdv = stats.stdev([Tiny_ResNet_RMSprop_Trials[i][13][1] for i in range(5)])

# ReLUConverganceValidationAccuracy = stats.mean([np.max(Tiny_ResNet_RMSprop_Trials[i][3]) for i in range(5)])
# ReLUConverganceValidationMedian = stats.median([np.max(Tiny_ResNet_RMSprop_Trials[i][3]) for i in range(5)])
# ReLUConverganceValidationStdv = stats.stdev([np.max(Tiny_ResNet_RMSprop_Trials[i][3]) for i in range(5)])

# ReLUConclusiveValidationAccuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][3][-1] for i in range(5)])
# ReLUStabilityIndex = ReLUConclusiveValidationAccuracy / ReLUConverganceValidationAccuracy

# ReLUConverganceEpochIndeces = [np.argmax(Tiny_ResNet_RMSprop_Trials[i][3]) for i in range(5)]
# ReLUConverganceEpochFloats = [float(ReLUConverganceEpochIndex) for ReLUConverganceEpochIndex in ReLUConverganceEpochIndeces]
# ReLUConverganceEpochAverage = stats.mean(ReLUConverganceEpochFloats)

# ReLUConverganceTrainingAccuracy = stats.mean([Tiny_ResNet_RMSprop_Trials[i][1][ReLUConverganceEpochIndeces[i]] for i in range(5)])

# ReLUFittingIndex = (100-ReLUConverganceTrainingAccuracy)/(100-ReLUTestingAccuracy)

# ReLUMinTrainingError = stats.mean([np.min(Tiny_ResNet_RMSprop_Trials[i][5]) for i in range(5)])

# ReLUMeanTrainingTime = stats.mean([stats.mean(Tiny_ResNet_RMSprop_Trials[i][9]) for i in range(5)])

# ReLUMeanValidationTime = stats.mean([stats.mean(Tiny_ResNet_RMSprop_Trials[i][11]) for i in range(5)])

# print(f"Average ReLU convergance Training accuracy: {ReLUConverganceTrainingAccuracy:.3f}")
# print(f"Average ReLU convergance Validation accuracy: {ReLUConverganceValidationAccuracy:.3f}")
# # print(f"Average ReLU convergance Testing Top-1 accuracy: {ReLUTestingAccuracy:.3f}")
# # print(f"Average ReLU convergance Testing Top-5 accuracy: {ReLUTesting5Accuracy:.3f}")
# print(f"Average ReLU conclusive Validation accuracy: {ReLUConclusiveValidationAccuracy:.3f}")
# #print(f"Median ReLU convergance Testing accuracy: {ReLUConverganceTestingMedian:.3f}")
# # print(f"Stdv of ReLU convergance Validation accuracy: {ReLUConverganceValidationStdv:.3f}")
# print(f"Average ReLU convergance epoch: {ReLUConverganceEpochAverage:.3f}")
# print(f"Average ReLU fitting index: {ReLUFittingIndex:.3f}")
# # print(f"Stdv of ReLU convergance test Top-1 accuracy: {ReLUConverganceTestingStdv:.3f}")
# # print(f"Stdv of ReLU convergance test Top-5 accuracy: {ReLUConvergance5TestingStdv:.3f}")
# #print(f"Average ReLU minimum training error: {ReLUMinTrainingError:.3f}")
# #print(f"Average ReLU epoch runtime: {ReLUMeanTrainingTime:.3f}")
# #print(f"Average ReLU testing runtime: {ReLUMeanTestingTime:.3f}")
# print()