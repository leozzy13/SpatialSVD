## Topic Alignment
library(purrr)
library(dplyr)
library(tidyr)
library(tibble)
library(stringr)
library(alto)
library(magrittr)

root_path = '~/Dropbox/SpLSI/data/stanford-crc/model'
root_path_Ahat = '~/Dropbox/SpLSI/data/stanford-crc/model/Ahats_splsi'
root_path_What = '~/Dropbox/SpLSI/data/stanford-crc/model/Whats_splsi'
label_data = read.csv('~/Dropbox/SpLSI/data/stanford-crc/dataset/Charville_c001_v001_r001_reg001.D.csv', row.names = 1)
cell_names = rownames(label_data)
immune_names = c('CD4 T cell', 'CD8 T cell', 'B cell', 'Macrophage', 'Granulocyte', 'Blood vessel', 'Stroma', 'Other')

setwd(root_path)
filenames_A = list.files(root_path_Ahat)
filenames_W = list.files(root_path_What)
ntopics = 6

get_K = function(x){
  v = seq(1:length(x))
  diff = x-v
  K = length(x)
  for(i in 1:length(x)){
    if(diff[i]!=0){
      K=x[i]
      break
    }
  }
  return(K)
}
get_paths = function(method){
  models = list()
  filesA = filenames_A[grep(method, filenames_A)]
  if(method=="PLSI"){filesA = filesA[!grepl("SPLSI", filesA)]}
  filesW = filenames_W[grep(method, filenames_W)]
  if(method=="PLSI"){filesW = filesW[!grepl("SPLSI", filesW)]}
  for(t in 1:ntopics){
    topic_name = as.character(t)
    print(topic_name)
    Ahat = read.csv(paste0(root_path_Ahat, '/', filenames_A[t]), header = FALSE)
    Ahat = t(Ahat)
    rownames(Ahat) = immune_names
    What = read.csv(paste0(root_path_What, '/', filenames_W[t]), header = FALSE)
    #What = What[indices_3,]
    What = as.matrix(What)
    models[[topic_name]] <- list(gamma = What, beta = t(as.matrix(Ahat)))
  }
  result = align_topics(models)
  paths = compute_number_of_paths(result, plot = FALSE)$n_paths
  K_hat = get_K(paths)
  print(paste0("K_hat is ", K_hat))
  p = plot(result)
  print(p)
}

get_paths('splsi')

par(mfrow=c(1,2))

methods = c("SPLSI", "PLSI", "SLDA")
for(m in methods){
  get_paths(m)
}


setwd('~/Desktop/SpLSI/utils')
source('align_topics.R')
source('reorder.R')
spleen_D = read.csv("~/Desktop/SpLSI/spleen_D.csv", row.names=1)
ntopics = 10
lda_params <- setNames(map(1:ntopics, ~ list(k = .)), 1:ntopics)
lda_models <- run_lda_models(spleen_D, lda_params)
result <- align_topics(lda_models)
compute_number_of_paths(result)
plot(result)

method = "product"
.check_align_input(lda_models, method)
weight_fun <- ifelse(method == "product", product_weights, transport_weights)
if (is.null(names(lda_models))) { names(lda_models) <- seq_along(lda_models) }
topics = topics_list(lda_models)
weights <-
  align_graph(
    edges = setup_edges("all", names(lda_models)),
    gamma_hats = map(lda_models, ~ .$gamma),
    beta_hats = map(lda_models, ~ exp(.$beta)),
    weight_fun = weight_fun
  )

paths <- topics %>%
  filter(m == 7) %>%
  mutate(path = k) %>%
  select(m, k, path)

model_names = c(1,2,3,4,5,6,7)
for (model in rev(model_names)[-1]) {
  paths_m <- weights %>%
    filter(m == model) %>%
    mutate(match_weight = 0.5 * fw_weight + 0.5 * bw_weight) %>%
    group_by(k) %>%
    slice_max(match_weight) %>%
    left_join(paths, by = c("k_next" = "k", "m_next" = "m")) %>%
    select(m, k, path)
  paths <- bind_rows(paths, paths_m)
}

paths <- paths %>%
  arrange(m, k) %>%
  mutate(path = factor(path, levels = sort(unique(path))))
