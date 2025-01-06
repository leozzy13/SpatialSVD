library(ggplot2)
library(RColorBrewer)
crc_K_results <- read_csv("data/stanford-crc/crc_K_results.csv")

line_color = hcl.pals()(n = 3, name = "Set3")
line_color = c("#E78AC3", "#8DA0CB", "#A6D854")
crc_K_summ = crc_K_results %>%
  group_by(K) %>%
  summarize(
    med1 = median(l1_dist, na.rm = TRUE),
    sd1 = IQR(l1_dist, na.rm = TRUE)/2,
    med2 = median(cos_sim, na.rm = TRUE),
    sd2 = IQR(cos_sim, na.rm = TRUE)/2,
    med3 = median(cos_sim_ratio, na.rm = TRUE),
    sd3 = IQR(cos_sim_ratio, na.rm = TRUE)/2,
  )

p1 = ggplot(data = crc_K_summ,
       aes(x=K, y=med1)) +
  geom_line(color = line_color[1]) +
  geom_point(color = line_color[1]) +
  geom_errorbar(aes(ymin = med1-sd1, ymax = med1+sd1), width = 0.2,
                color = line_color[1]) +
  labs(
    title = NULL,
    x = "K",
    y = "L1 Distance",
  ) +
  theme_minimal()

p2 = ggplot(data = crc_K_summ,
            aes(x=K, y=med2)) +
  geom_line(color = line_color[2]) +
  geom_point(color = line_color[2]) +
  geom_errorbar(aes(ymin = med2-sd2, ymax = med2+sd2), width = 0.2,
                color = line_color[2]) +
  labs(
    title = NULL,
    x = "K",
    y = "Cosine Similarity",
  ) +
  theme_minimal()

p3 = ggplot(data = crc_K_summ,
            aes(x=K, y=med3)) +
  geom_line(color = line_color[3]) +
  geom_point(color = line_color[3]) +
  geom_errorbar(aes(ymin = med3-sd3, ymax = med3+sd3), width = 0.2,
                color = line_color[3]) +
  labs(
    title = NULL,
    x = "K",
    y = "Relative Cosine Similarity",
  ) +
  theme_minimal()

grid.arrange(p1, p2, p3, ncol=3)
