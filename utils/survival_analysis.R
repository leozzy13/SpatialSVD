library(compositions)

root_path = '~/Dropbox/SpLSI'
setwd(root_path)

survival = read_csv('survival.csv')
colnames(survival) = 
  c("region_id", "Topic1","Topic2","Topic3","Topic4",
    "Topic5","Topic6","primary_outcome", "recurrence",
    "length_of_disease_free_survival")

# plot
long_df <- survival %>%
  pivot_longer(
    cols = starts_with("Topic"),
    names_to = "Topic",
    values_to = "Proportion"
  ) #%>%
  #mutate(Topic = str_replace(Topic, "Topic(\\d+)_prop", "Topic \\1"))
library(tidyverse)
long_df_clean <- long_df %>%
  filter(!is.na(recurrence))

p1 = ggplot(long_df_clean, aes(x = Topic, y = Proportion, fill = as.factor(primary_outcome))) +
  geom_boxplot() +
  labs(
    title = NULL,
    x = NULL,
    y = "Topic Proportion",
    fill = "primary_outcome"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")+
  scale_fill_brewer(palette = "Pastel1") # Optional: Use a color palette


# logistic regression
X = scale(ilr(survival[,2:7]), center = TRUE, scale=FALSE)
y = survival$primary_outcome
fit = glm(y ~ X, family=binomial)
summary(fit)
ilr2clr(coef(fit)[-1],x=X)

# survival analysis
library(ggfortify)
time = survival$length_of_disease_free_survival
status = survival$primary_outcome
cox = coxph(Surv(time, status) ~ X)
ilr2clr(coef)
summary(cox)
ilr2clr(coef(cox),x=X)
#plot(cox_fit, main = "cph model", xlab="Days")
autoplot(cox_fit)

aa_fit <-aareg(Surv(time, status) ~ 
                 survival$Topic1 +)
autoplot(aa_fit)
