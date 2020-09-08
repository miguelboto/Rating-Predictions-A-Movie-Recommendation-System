
#Create train and validation sets
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

head(edx, 5)
head(validation, 5)

#Include two new columns for year of released and year of rating

edx2 <- mutate(edx, year_rated = year(as_datetime(timestamp)),
               year = as.numeric(str_sub(title,-5,-2)))

validation2 <- mutate(edx, year_rated = year(as_datetime(timestamp)),
                      year = as.numeric(str_sub(title,-5,-2)))
head(edx2, 5)
head(validation2, 5)


###########################
#Analysis & Visualizations#
###########################


#Movilens database has the following composition:
head(edx2, 5) 

#Data summary:                      
summary(edx2) %>%
  knitr::kable()

#Summary resume:

edx2 %>% summarize(
  'Unique Users'=n_distinct(userId),
  Movies=n_distinct(movieId),
  'Minumum Rating'=min(rating),
  'Maximum Rating'=max(rating) 
) %>%
  knitr::kable()


#Rating distribution
edx2 %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, fill = "blue") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")

#Ratings per movie
edx2 %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=25, color= "black", fill="orange")+
  scale_x_log10()+
  xlab("Ratings")+
  ylab("Movies")+
  ggtitle("Ratings per Movie")

#Ratings per user
edx2 %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=25, color= "black", fill="brown")+
  scale_x_log10()+
  xlab("Ratings")+
  ylab("Users")+
  ggtitle("Ratings per User")


#Mean movies ratings per year
edx2 %>%
  group_by(year) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(year, avg)) +
  geom_point(color = "brown3")+
  scale_y_discrete(limits = c(seq(0.5,5,0.5)))+
  geom_smooth()+
  xlab("Average Ratings")+
  ylab("Year")+
  ggtitle("Mean Movies Ratings per Year")


#Movies per year of released

edx2 %>%
  group_by(year) %>%
  count(movieId) %>%
  summarize(total = sum(n))%>%
  ggplot(aes(year, total))+
  geom_line(aes(color="red"))+
  xlab("Year")+
  ylab("Number of Movies")+
  ggtitle("Movies per year")


#Mean Rating per Rated Year and Released Year
ggplot() +  
  geom_point(data = edx2 %>% 
               group_by(year) %>%
               summarize(avg = mean(rating), number = n()),
             aes(year, avg, col="year"))+
  geom_point(data = edx2 %>% 
               group_by(year_rated) %>%
               summarize(avg = mean(rating), number = n()),
             aes(year_rated, avg, col="year_rated")) +
  ggtitle("Mean Rating per Rated Year and Released Year")+
  scale_y_discrete(limits = c(seq(0.5,5,0.5))) +
  xlab("Year")+
  ylab("Rate")

#separate genres in rows:
sep_genres <- edx2 %>% separate_rows(genres, sep ="\\|")


num_genres <- sep_genres %>% group_by(genres) %>% 
  summarize(totalratings = n(), averageRating = mean(rating)) %>%
  arrange(desc(totaltatings))

#Average rating  per genre
num_genres %>%  
  mutate(genres = reorder(genres, totalratings)) %>%
  ggplot(aes(x= genres, y = averagerating, color = genres)) +
  geom_point(aes(size=totalratings)) +
  ggtitle("Average Rating per Genre") + 
  xlab("Genre")+
  ylab("Rating") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  guides(size=FALSE)
  


#########################
#Predictions and results#
#########################

#Simple Mean Ratings prediction

mu <- mean(edx2$rating)
mu  

naive_RMSE <- RMSE(validation2$rating, mu)
naive_RMSE

RMSE_results = data.frame(Method = "Naive by Mean", RMSE = naive_RMSE) 
RMSE_results  

#Bias Movie Effects Model 

avgs <- edx2 %>% group_by(movieId) %>%
  summarise(bi = mean(rating - mu))


predictions <- mu + validation2 %>%
  left_join(avgs, by="movieId") %>%
  pull(bi)

RMSE_effect_model<- RMSE(predictions, validation2$rating)
RMSE_effect_model

RMSE_results <-  bind_rows(RMSE_results, 
                           data.frame(Method = "Effect Model", 
                                      RMSE = RMSE_effect_model))
RMSE_results %>% knitr::kable()

#Bias Movies & users Effects Model  

avgs_us <- edx2 %>% group_by(userId) %>%
  left_join(avgs, by = "movieId") %>%
  summarise(bu = mean(rating - mu - bi))

avgs_us

predictions_2 <- validation2 %>%
  left_join(avgs, by = "movieId") %>%
  left_join(avgs_us, by = "userId") %>%
  mutate(pre_bi_bu = mu + bi + bu) %>%
  pull(pre_bi_bu)

RMSE_usermovie_effect_model <- RMSE(predictions_2, validation2$rating)

RMSE_usermovie_effect_model

RMSE_results <- bind_rows(RMSE_results, 
                          data.frame(Method="User & Movie Effect Model", 
                                     RMSE = RMSE_usermovie_effect_model))

RMSE_results %>% knitr::kable()

#Regularization

lambdas <- seq(0, 10, 0.25)

RMSES <- sapply(lambdas, function(l){
  
  mu <- mean(edx2$rating)
  
  bi <- edx2 %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(n()+l))
  bu <- edx2 %>%
    left_join(bi, by="movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - bi - mu)/(n()+l))
  predicted_ratings <- validation2 %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    mutate(pred = mu + bi + bu) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation2$rating))
})
qplot(lambdas, RMSES)
lambdas[which.min(RMSES)]

#Final Model

RMSE_results <- bind_rows(RMSE_results, 
                          data_frame(Method="Regularized", 
    
                                    RMSE = min(RMSES)))

#Summary Table

RMSE_results %>% knitr::kable()



