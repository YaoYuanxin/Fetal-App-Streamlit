library(caret)
library(nlme)
library(readr)
library("RSQLite")

# connect to the sqlite file
sqlite    <- dbDriver("SQLite")
sequential_db <- dbConnect(sqlite,"sequential.db")
tables <- dbListTables(sequential_db)


## exclude sqlite_sequence (contains table information)
#tables <- tables[tables != "sqlite_sequence"]

lDataFrames <- vector("list", length=length(tables))

## create a data.frame for each table
for (i in seq(along=tables)) {
  lDataFrames[[i]] <- dbGetQuery(conn=sequential_db, statement=paste("SELECT * FROM '", tables[[i]], "'", sep=""))
}

df <- as.data.frame(lDataFrames[[1]])

df_grouped = groupedData( efw ~ gadays | id, data = df)
gadays = df_grouped$gadays
gadays_2 = gadays^2


df_grouped_2 = as.data.frame(cbind(df_grouped,gadays_2))

st = getInitial(efw~SSlogis(gadays,Asym,xmid,scal),data=df_grouped)
st2 = lm(efw~gadays+gadays_2,data=df_grouped_2)$coefficients

co = nlmeControl( returnObject=T)

fit3 = nlme(efw~eff,fixed=eff~gadays+gadays_2,random=eff~gadays+gadays_2|id,
            data=df_grouped_2,start=st2,control=co)

augmented_time = seq(84,301)
new_data = data.frame(id = rep(unique(df_grouped_2$id),each = length(augmented_time)), 
                        gadays = rep(augmented_time,length(unique(unique(df_grouped_2$id)))),
                        gadays_2 = rep(augmented_time,length(unique(unique(df_grouped_2$id))))^2)

augmented_quad = predict(fit3, newdata = new_data , level = 1)
augmented_quad_df = data.frame(id = rep(unique(df_grouped_2$id),each = length(augmented_time)), 
                               gadays = rep(augmented_time,length(unique(unique(df_grouped_2$id)))),
                               gadays_2 = rep(augmented_time,length(unique(unique(df_grouped_2$id))))^2,
                               efw = augmented_quad)


dbWriteTable(conn = sequential_db, 
             name = "augmented_quad_df", 
             value = augmented_quad_df,
             overwrite = TRUE,
             temporary = FALSE) 

#dbDisconnect(sequential_db)

