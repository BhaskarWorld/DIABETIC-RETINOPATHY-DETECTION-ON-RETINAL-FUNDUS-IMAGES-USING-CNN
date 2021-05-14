import csv
import shutil
import os
# Categorizes the images according to their label mentioned in the provided csv file
filename = "trainLabels.csv"
rows = []

# reading csv file
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        rows.append(row)


for row in rows:
    if(row[1] == '0'):
        shutil.move('H:/project/train_data/' +
                    row[0]+'.jpeg', 'H:/project/CategorizedData/class_0/'+row[0]+'.jpeg')
    elif(row[1] == '1'):
        shutil.move('H:/project/train_data/' +
                    row[0]+'.jpeg', 'H:/project/CategorizedData/class_1/'+row[0]+'.jpeg')
    elif(row[1] == '2'):
        shutil.move('H:/project/train_data/' +
                    row[0]+'.jpeg', 'H:/project/CategorizedData/class_2/'+row[0]+'.jpeg')
    elif(row[1] == '3'):
        shutil.move('H:/project/train_data/' +
                    row[0]+'.jpeg', 'H:/project/CategorizedData/class_3/'+row[0]+'.jpeg')
    elif(row[1] == '4'):
        shutil.move('H:/project/train_data/' +
                    row[0]+'.jpeg', 'H:/project/CategorizedData/class_4/'+row[0]+'.jpeg')
