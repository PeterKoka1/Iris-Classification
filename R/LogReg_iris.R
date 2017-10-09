attach(iris)
set.seed(123)
library(caret)

###: Predict versicolor and setosa
iris<-iris[1:100,]
train<-createDataPartition(iris$Species, list=FALSE)
train.X<-iris[train,-5]
train.X
train.y<-iris[train,5]
test.X<-iris[-train,-5]
test.y<-iris[-train,5]

cor(iris[,-5])
pairs(iris[,-5], main="Iris Data", pch=21,
      bg=c("Red","Blue","Green")[unclass(iris$Species)])
library(ggplot2);library(GGally)
ggpairs(iris[train,-5])
# petal width and petal length have strong correlation
lm.fit.petals<-lm(Petal.Length~Petal.Width,data=iris)
summary(lm.fit.petals) # t-stat 43.39 and p-value near 0

# Nearly all pairs or groupings of preds don't allow algorithm to converge
glm.fit<-glm(iris$Species ~ Sepal.Width,
             data=iris,family=binomial,subset=train)
summary(glm.fit)
summary(glm.fit)$coef[2,4] # p-value of 0.00518
par(mfrow=c(2,2))
plot(glm.fit)
contrasts(test.y)
glm.probs<-predict(glm.fit,test.X,type="response")
glm.preds<-rep("setosa",length(test.y))
glm.preds[glm.probs>.5]="versicolor"
table(glm.preds,test.y)
mean(glm.preds==test.y) # 82.0% accuracy