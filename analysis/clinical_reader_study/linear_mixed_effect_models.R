library(lme4)
library(lmerTest) #install.packages("lmerTest")

df <- read.csv("~/Downloads/collated_reader_study_results.csv", header = TRUE)
removed_id <- c(29,73,105,122,141,143,152,166,184)
df <- df[!df$global_set_id %in% removed_id,]

df$source <- factor(df$source, levels = c("original_history","additional_history","llm_indication_claude","llm_indication_qwen"))

model_com <- lmer(comprehensiveness ~ source+(1|global_set_id)+(1|user_id), data=df)
summary(model_com)

model_fac <- lmer(factuality ~ source+(1|global_set_id)+(1|user_id), data=df)
summary(model_fac)

model_con <- lmer(conciseness ~ source+(1|global_set_id)+(1|user_id), data=df)
summary(model_con)


#post hoc testing

library(emmeans)
emm_comp <- emmeans(model_com, ~source)
pairs(emm_comp,adjust="tukey")

emm_comp <- emmeans(model_fac, ~source)
pairs(emm_comp,adjust="tukey")

emm_comp <- emmeans(model_con, ~source)
pairs(emm_comp,adjust="tukey")


#rank data assessment
library(ordinal) #install.package("ordinal")
library(tidyr)
library(dplyr)

df_collapse = df[seq(4, nrow(df), by=4), ]
df_long = pivot_longer(df_collapse, cols=starts_with("factor_ranking"), names_to="rank", values_to="factor")
df_long$rank[df_long$rank == "factor_ranking_1"] <-"1"
df_long$rank[df_long$rank == "factor_ranking_2"] <-"2"
df_long$rank[df_long$rank == "factor_ranking_3"] <-"3"

df_long <- df_long %>% mutate(rank = as.factor(rank))

model <- clmm(rank ~ factor +(1|global_set_id)+(1|user_id),data=df_long)
summary(model)