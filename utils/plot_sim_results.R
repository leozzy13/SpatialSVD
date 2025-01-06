library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyverse)
library(gridExtra)
library(vroom)

setwd("/Users/jeong-yeojin/Desktop/results_final")
files = list.files("/Users/jeong-yeojin/Desktop/results_final")
res_df = vroom(files)
write.csv(res_df, "results_all.csv")
res_df= read.csv("/Users/jeong-yeojin/Desktop/results_/results_all2.csv")
res_df$K = factor(res_df$K)

##### L2 Error #####
# Reshape to long
res_df_long <- res_df %>%
  pivot_longer(cols = c(plsi_err, hooi_err, ts_err, lda_err, slda_err),
               names_to = "method_error", values_to = "error") %>%
  mutate(Method = case_when(
    str_detect(method_error, "^plsi_err$") ~ "PLSI",
    #str_detect(method_error, "^splsi_err$") ~ "SPLSI",
    str_detect(method_error, "^hooi_err$") ~ "SPLSI",
    str_detect(method_error, "^ts_err$") ~ "TopicSCORE",
    str_detect(method_error, "^lda_err$") ~ "LDA",
    str_detect(method_error, "^slda_err$") ~ "SLDA"
  )) 
res_df_long$Method <- factor(res_df_long$Method, levels = c("SPLSI", "PLSI", "HOOI","TopicSCORE", "LDA", "SLDA"))

# check outlier
# Plot 1
res_df_long1 =  res_df_long %>%
  filter(K==3, n==1000) %>%
  mutate(error = error / n) %>%
  group_by(N, p, Method) %>%
  summarize(
    mean_error = median(error, na.rm = TRUE),
    sd_error = IQR(error, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p.labs <- paste("p =", c(20,30,50,100,200,500))
names(p.labs) <- unique(res_df_long1$p)
p1 = ggplot(res_df_long1, aes(x = N, 
                        y = mean_error, 
                        color = Method)) +
  geom_point() +
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  facet_wrap(~ p, labeller = labeller(p = as_labeller(p.labs)), ncol = 3) +
  labs(x = "N", y = "l2 Error") +
  theme_bw() +
  scale_x_log10()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  #theme(legend.position = "none")

# Plot 2
res_df_long2 =  res_df_long %>%
  filter(K == 3, p == 30, N == 30) %>%
  mutate(adjusted_error = error / n) %>%
  group_by(n, Method) %>%
  summarize(
    mean_error = median(adjusted_error, na.rm = TRUE), 
    sd_error = IQR(adjusted_error, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p2 = ggplot(res_df_long2, aes(x = n, 
                         y = mean_error, 
                         color = Method)) +
  geom_point() +
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  labs(x = "n", y = "l2 Error") +
  scale_x_log10()+
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(legend.position = "none")



# Plot 3
res_df_long3 =  res_df_long %>%
  filter(N == 30, p == 30, n ==1000) %>%
  mutate(error = error/n) %>%
  group_by(K, Method) %>%
  summarize(
    mean_error = median(error, na.rm = TRUE),
    sd_error = IQR(error, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p3 = ggplot(res_df_long3, aes(x = K, 
                              y = mean_error, 
                              color = Method,
                              group = Method)) +
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  labs(x = "K", y = "l2 Error") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))+
  theme(legend.position = "none")

grid.arrange(p2, p3, ncol = 2)




##### L1 Error #####
# Reshape to long
res_df_long_l1 <- res_df %>%
  pivot_longer(cols = c(plsi_l1_err, hooi_l1_err, ts_l1_err, lda_l1_err, slda_l1_err),
               names_to = "method_error", values_to = "error") %>%
  mutate(Method = case_when(
    str_detect(method_error, "^plsi_l1_err$") ~ "PLSI",
    #str_detect(method_error, "^splsi_l1_err$") ~ "SPLSI",
    str_detect(method_error, "^hooi_l1_err$") ~ "SPLSI",
    str_detect(method_error, "^ts_l1_err$") ~ "TopicSCORE",
    str_detect(method_error, "^lda_l1_err$") ~ "LDA",
    str_detect(method_error, "^slda_l1_err$") ~ "SLDA"
  )) 
res_df_long_l1$Method <- factor(res_df_long_l1$Method, levels = c("SPLSI", "PLSI", "TopicSCORE", "LDA", "SLDA"))


# Plot 1
res_df_long4 =  res_df_long_l1 %>%
  filter(K==3, n==1000) %>%
  mutate(error = error / n) %>%
  group_by(N, p, Method) %>%
  summarize(
    mean_error = median(error, na.rm = TRUE),
    sd_error = IQR(error, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p.labs <- paste("p =", c(20,30,50,100,200,500))
names(p.labs) <- unique(res_df_long4$p)
p4 = ggplot(res_df_long4, aes(x = N, 
                              y = mean_error, 
                              color = Method)) +
  geom_point() +
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  facet_wrap(~ p, labeller = labeller(p = as_labeller(p.labs)), ncol = 3) +
  labs(x = "N", y = "l1 Error") +
  theme_bw() +
  scale_x_log10()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  #theme(legend.position = "none")

# Plot 2
res_df_long5 =  res_df_long_l1 %>%
  filter(K == 3, p == 30, N == 30) %>%
  mutate(adjusted_error = error / n) %>%
  group_by(n, Method) %>%
  summarize(
    mean_error = median(adjusted_error, na.rm = TRUE), 
    sd_error = IQR(adjusted_error, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p5 = ggplot(res_df_long5, aes(x = n, 
                              y = mean_error, 
                              color = Method)) +
  geom_point() +
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  labs(x = "n", y = "l1 Error") +
  scale_x_log10()+
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(legend.position = "none")



# Plot 3
res_df_long6 =  res_df_long_l1 %>%
  filter(N == 30, p == 30, n ==1000) %>%
  mutate(error = error / n) %>%
  group_by(K, Method) %>%
  summarize(
    mean_error = median(error, na.rm = TRUE),
    sd_error = IQR(error, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p6 = ggplot(res_df_long6, aes(x = K, 
                              y = mean_error, 
                              color = Method,
                              group = Method)) +
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  labs(x = "K", y = "l1 Error") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))+
  theme(legend.position = "none")

grid.arrange(p5, p6, ncol = 2)



##### Time #####
# Reshape to long
res_df_long_time <- res_df %>%
  pivot_longer(cols = c(plsi_time, hooi_time, ts_time, lda_time, slda_time),
               names_to = "method_time", values_to = "time") %>%
  mutate(Method = case_when(
    str_detect(method_time, "^plsi_time$") ~ "PLSI",
    #str_detect(method_time, "^splsi_time$") ~ "SPLSI",
    str_detect(method_time, "^hooi_time$") ~ "SPLSI",
    str_detect(method_time, "^ts_time$") ~ "TopicSCORE",
    str_detect(method_time, "^lda_time$") ~ "LDA",
    str_detect(method_time, "^slda_time$") ~ "SLDA"
  )) 
res_df_long_time$Method <- factor(res_df_long_time$Method, levels = c("SPLSI", "PLSI", "TopicSCORE", "LDA", "SLDA"))


# Plot 4
res_df_long_time_1 =  res_df_long_time %>%
  filter(K==3, n==1000) %>%
  group_by(N, p, Method) %>%
  summarize(
    mean_error = median(time, na.rm = TRUE),
    sd_error = IQR(time, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p.labs <- paste("p =", c(20,30,50,100,200,500))
names(p.labs) <- unique(res_df_long_time_1$p)
p7 = ggplot(res_df_long_time_1, aes(x = N, 
                              y = mean_error, 
                              color = Method)) +
  geom_point() +
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  facet_wrap(~ p, labeller = labeller(p = as_labeller(p.labs)), ncol = 3) +
  labs(x = "N", y = "Time") +
  theme_bw() +
  scale_x_log10()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  #theme(legend.position = "none")

# Plot 2
res_df_long_time_2 =  res_df_long_time %>%
  filter(K == 3, p == 30, N == 30) %>%
  group_by(n, Method) %>%
  summarize(
    mean_error = median(time, na.rm = TRUE), 
    sd_error = IQR(time, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p8 = ggplot(res_df_long_time_2, aes(x = n, 
                              y = mean_error, 
                              color = Method)) +
  geom_point() +
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  labs(x = "n", y = "Time") +
  scale_x_log10()+
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(legend.position = "none")



# Plot 3
res_df_long_time3 =  res_df_long_time %>%
  filter(N == 30, p == 30, n ==1000) %>%
  group_by(K, Method) %>%
  summarize(
    mean_error = median(time, na.rm = TRUE),
    sd_error = IQR(time, na.rm = TRUE)/2,
    .groups = 'drop'
  )

p9 = ggplot(res_df_long_time3, aes(x = K, 
                              y = mean_error, 
                              color = Method,
                              group = Method)) +
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin = mean_error-sd_error, ymax = mean_error+sd_error), width = 0.2) +
  labs(x = "K", y = "Time") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1))+
  theme(legend.position = "none")

grid.arrange(p8, p9, ncol = 2)

grid.arrange(p4, p1, p7, ncol = 3)
#grid.arrange(p2, p3, p5, p6, p8, p9, ncol = 2)
grid.arrange(p5, p2, p8, 
             p6, p3, p9, ncol = 3)

library(cowplot)
plots <- list(p5, p2, p8, p6, p3, p9)
legend <- get_legend(p5 + theme(legend.position = "right"))
plot_grid <- plot_grid(plotlist = lapply(plots, function(x) x + theme(legend.position = "none")), ncol = 3)
final_plot <- plot_grid(plot_grid, legend, ncol = 2, rel_widths = c(3, 0.4))
final_plot

K = 2
A = matrix(runif(100),10,10)
A[,1] <- abs(A[,1])
R <- apply(A[, 2:K, drop = FALSE],2,function(x) x/A[,1])
