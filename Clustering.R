df <- read.csv("E:/Fastract/Semester 4/Image Processing/Project/data.csv")
View(df)

#Set Clustering
set.seed(846325)
group <- kmeans(df[, c("Distance", "Area")], centers=7, nstart=10)
group

library("ggplot2", lib.loc="~/R/win-library/3.4")
ggplot(df, aes(x=df$Distance, y=df$Area, colour=  group$cluster)) + geom_point()

library(factoextra)
fviz_cluster(group, df[,0:2], stand=FALSE, geom="point")
             