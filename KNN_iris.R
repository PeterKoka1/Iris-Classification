?iris
names(iris)
dim(iris) # 150 5

summary(iris) # note: regularize
attach(iris)
cor(iris[,-5])

pairs(iris[,-5], main="Iris Data", 
      pch=21, bg=c("Red","Blue","Green")[unclass(iris$Species)])
plot(iris$Sepal.Length,iris$Sepal.Width, pch=21, bg=c("Red","Blue","Green"))
# linearity between species

###: All predictors
standardized.X<-scale(iris[,-5])
set.seed(123)
library(caret)
train<-createDataPartition(iris$Species, list=FALSE)
train.X<-standardized.X[train,-5]
train.y<-iris[train,5]
test.X<-standardized.X[-train,-5]
test.y<-iris[-train,5]

###: K=1
knn.pred1<-knn(train.X,test.X,train.y,k=1)
table(knn.pred,test.y)
mean(knn.pred!=test.y) # test error of 6.67%
###: K=3
knn.pred3<-knn(train.X,test.X,train.y,k=3)
mean(knn.pred!=test.y) # test error of 6.67% again
###: K=5
knn.pred5<-knn(train.X,test.X,train.y,k=5)
mean(knn.pred!=test.y) # test error of 5.33%

#: k-nn with k=5: ~95.7% accuracy

versicolor<- 1 - (25 / (25 + 0 + 4)) # 13.8% error rate (all errors from versicolor)

###: Petal.Length and Petal.Width
plot(Petal.Length,Petal.Width,main="Petal Length vs. Petal Width",
     pch=21,bg=c("Red","Blue","Green")[unclass(Species)])
knn.pred<-knn(train.X[,c(3,4)],test.X[,c(3,4)],train.y,k=1)
table(knn.pred,test.y)
mean(knn.pred==test.y) # k=1: ~97.3% accuracy
knn.pred<-knn(train.X[,c(3,4)],test.X[,c(3,4)],train.y,k=3)
table(knn.pred,test.y)
mean(knn.pred==test.y) # k=3: ~94.67% accuracy, less flexible model performed worse

###: Sepal.Length and Sepal.Width (least-linear predictors)
plot(Sepal.Length,Sepal.Width,main="Sepal Length vs. Sepal Width",
     pch=21,bg=c("Red","Blue","Green")[unclass(Species)])
knn.pred<-knn(train.X[,c(1,2)],test.X[,c(1,2)],train.y,k=1)
table(knn.pred,test.y)
mean(knn.pred==test.y) # k=1: ~70.7% accuracy
knn.pred<-knn(train.X[,c(1,2)],test.X[,c(1,2)],train.y,k=5)
mean(knn.pred==test.y) # k=5 had best accuracy ~ 77.3% -> less flexible model outperforms
# k>5 increase in error rate