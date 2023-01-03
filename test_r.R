library(caret)
#library(pROC)
#library(epiR)
#library(ResourceSelection)

library(nlme)

library(readr)
library("RSQLite")

# connect to the sqlite file
sqlite    <- dbDriver("SQLite")
exampledb <- dbConnect(sqlite,"fetal_app_database.db")

dbListTables(exampledb)

## list all tables
tables <- dbListTables(exampledb)

## exclude sqlite_sequence (contains table information)
tables <- tables[tables != "sqlite_sequence"]

lDataFrames <- vector("list", length=length(tables))

## create a data.frame for each table
for (i in seq(along=tables)) {
  lDataFrames[[i]] <- dbGetQuery(conn=exampledb, statement=paste("SELECT * FROM '", tables[[i]], "'", sep=""))
}

as.data.frame(lDataFrames[[2]])

