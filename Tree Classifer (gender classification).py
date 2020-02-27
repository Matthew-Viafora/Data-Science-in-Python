from sklearn import tree

#[height(cm), wieght(kg), shoe size(US)]

x = [[181,80,12],[177,70,6],[160,60,7],[154,54,4],
    [166, 65,13],[190,90,11],[175,64,8],[177,70,7],[159,55,10],
    [171,75,8],[181,85,10]]

y = ["male","female","female","female","male","male",
    "male","female","male","female","male"]


#intialize tree classifier 

clf = tree.DecisionTreeClassifier()

#fit with training data
clf = clf.fit(x,y)

#make prediction

prediction = clf.predict([[80,90,4]])

print (prediction)








