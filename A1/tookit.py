import torch
#this programe is used to store the commonly used functions I wrote

#     #torch.max()function return the biggest value of each row or column. if the 
#     # second argument is 0,group by column;if the second argument is 1,group by row
#     # so torch.max(predict_solution, 1) return a 
def precision(predict_solution,labels):
    pred = torch.max(predict_solution.data, 1)[1]
    number_of_right=0
    for label,value in enumerate(pred):
        if value==labels[label]:
            number_of_right+=1
    return number_of_right/len(labels)