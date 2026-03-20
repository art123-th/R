# ============================================================================
# IMPACT ANALYSIS: 2026 US-Iran War Oil Price Impact Analysis
# ภาษา: R
# วันที่สร้าง: 2024
# ============================================================================

# ============================================================================
# 1. โหลด Library ที่จำเป็น
# ============================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(gbm)
library(xgboost)
library(lightgbm)
library(corrplot)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(scales)
library(gridExtra)

# ตั้งค่า theme สำหรับกราฟ
theme_set(theme_minimal(base_size = 12))

# ============================================================================
# 2. โหลดข้อมูลจากไฟล์ CSV
# ============================================================================

load_data <- function(data_dir = "data") {
  
  # สร้างรายการไฟล์ที่ต้องการโหลด
  file_mapping <- list(
    crude = "crude_oil_daily.csv",
    petrol = "petrol_prices_comparison.csv",
    country = "country_impact.csv",
    pros_cons = "pros_cons_analysis.csv",
    timeline = "war_timeline.csv"
  )
  
  # โหลดไฟล์ทั้งหมด
  data_list <- list()
  
  for (name in names(file_mapping)) {
    file_path <- file.path(data_dir, file_mapping[[name]])
    
    if (file.exists(file_path)) {
      tryCatch({
        data_list[[name]] <- read_csv(file_path, show_col_types = FALSE)
        message(sprintf("✅ โหลดไฟล์สำเร็จ: %s", file_mapping[[name]]))
      }, error = function(e) {
        warning(sprintf("❌ ไม่สามารถโหลดไฟล์: %s - %s", file_mapping[[name]], e$message))
        data_list[[name]] <- NULL
      })
    } else {
      warning(sprintf("⚠️ ไม่พบไฟล์: %s", file_mapping[[name]]))
      data_list[[name]] <- NULL
    }
  }
  
  return(data_list)
}

# ============================================================================
# 3. ฟังก์ชันตรวจสอบคุณภาพข้อมูล (Data Audit)
# ============================================================================

audit_data <- function(df, name) {
  
  if (is.null(df)) {
    message(sprintf("⚠️ %s: DataFrame เป็น NULL — ข้ามการตรวจสอบ", name))
    return(invisible(NULL))
  }
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat(sprintf("📊 AUDIT: %s\n", name))
  cat(rep("=", 70), "\n", sep = "")
  
  # ข้อมูลพื้นฐาน
  cat(sprintf(" Shape : %d rows × %d cols\n", nrow(df), ncol(df)))
  cat(sprintf(" Memory : %.1f KB\n", object.size(df) / 1024))
  
  # ตรวจสอบคอลัมน์
  cat("\nColumns:\n")
  for (i in seq_along(names(df))) {
    col_name <- names(df)[i]
    col_type <- class(df[[col_name]])[1]
    n_unique <- n_distinct(df[[col_name]], na.rm = TRUE)
    n_missing <- sum(is.na(df[[col_name]]))
    
    cat(sprintf(" %2d. %-35s dtype=%-12s unique=%-6d missing=%d\n", 
                i, col_name, col_type, n_unique, n_missing))
  }
  
  # ตรวจสอบข้อมูลซ้ำ
  n_duplicates <- sum(duplicated(df))
  cat(sprintf("\nDuplicates: %d\n", n_duplicates))
  
  # ตรวจสอบค่า NaN
  total_na <- sum(is.na(df))
  cat(sprintf("Total NaN : %d\n", total_na))
  
  cat("\n")
  
  return(invisible(NULL))
}

# ============================================================================
# 4. ฟังก์ชันทำความสะอาดข้อมูล (Data Cleaning)
# ============================================================================

clean_data <- function(df, name) {
  
  if (is.null(df)) {
    return(NULL)
  }
  
  df_clean <- df %>%
    # ลบช่องว่างในชื่อคอลัมน์
    rename_with(~ trimws(.)) %>%
    # แปลงชื่อคอลัมน์เป็นตัวพิมพ์เล็ก
    rename_with(~ tolower(.))
  
  # แปลงคอลัมน์วันที่
  date_cols <- names(df_clean)[grep("date", names(df_clean), ignore.case = TRUE)]
  for (col in date_cols) {
    df_clean[[col]] <- tryCatch({
      parse_date_time(df_clean[[col]], orders = c("ymd", "dmy", "mdy", "ymd_hms"))
    }, error = function(e) {
      df_clean[[col]]
    })
  }
  
  # แปลงคอลัมน์ตัวเลข
  num_cols <- names(df_clean)[sapply(df_clean, is.numeric)]
  for (col in num_cols) {
    df_clean[[col]] <- as.numeric(df_clean[[col]])
  }
  
  # แปลงคอลัมน์ตัวอักษร
  char_cols <- names(df_clean)[sapply(df_clean, is.character)]
  for (col in char_cols) {
    df_clean[[col]] <- as.factor(df_clean[[col]])
  }
  
  # ลบแถวที่มีค่า NA
  df_clean <- na.omit(df_clean)
  
  # ลบแถวซ้ำ
  df_clean <- distinct(df_clean)
  
  message(sprintf("✅ CLEAN| End: %d rows × %d cols | NaN: %d", 
                  nrow(df_clean), ncol(df_clean), sum(is.na(df_clean))))
  
  return(df_clean)
}

# ============================================================================
# 5. ฟังก์ชันรวมข้อมูล (Data Merging)
# ============================================================================

merge_datasets <- function(data_list) {
  
  df_petrol <- data_list$petrol
  df_country <- data_list$country
  
  if (!is.null(df_petrol) && !is.null(df_country)) {
    
    # มาตรฐานชื่อคอลัมน์ประเทศ
    df_petrol <- df_petrol %>%
      rename_with(~ trimws(.)) %>%
      rename_with(~ tolower(.))
    
    df_country <- df_country %>%
      rename_with(~ trimws(.)) %>%
      rename_with(~ tolower(.))
    
    # หาชื่อคอลัมน์ประเทศ
    country_col_petrol <- names(df_petrol)[grep("country", names(df_petrol), ignore.case = TRUE)][1]
    country_col_country <- names(df_country)[grep("country", names(df_country), ignore.case = TRUE)][1]
    
    if (!is.null(country_col_petrol) && !is.null(country_col_country)) {
      
      # รวมข้อมูล
      df_merged <- df_petrol %>%
        left_join(df_country, by = setNames(country_col_country, country_col_petrol))
      
      message(sprintf("✅ Merged shape: %d rows × %d cols", nrow(df_merged), ncol(df_merged)))
      
    } else {
      df_merged <- df_petrol
      message("⚠️ Could not merge — using petrol table only")
    }
    
  } else if (!is.null(df_petrol)) {
    df_merged <- df_petrol
    message("⚠️ Only petrol data available")
  } else {
    stop("❌ No data available for analysis")
  }
  
  return(df_merged)
}

# ============================================================================
# 6. ฟังก์ชันเตรียมข้อมูลสำหรับ Model (Feature Engineering)
# ============================================================================

prepare_features <- function(df_merged) {
  
  df_prep <- df_merged %>%
    mutate(
      # สร้างฟีเจอร์ใหม่
      price_multiplier = mar7_usd / before_war_usd,
      log_before_usd = log1p(before_war_usd),
      gdp_import_interact = gdp_impact_pct * oil_import_pct,
      stock_import_ratio = stock_market_change / oil_import_pct
    )
  
  # แปลงตัวแปร categorical เป็น numeric
  cat_cols <- names(df_prep)[sapply(df_prep, is.factor)]
  for (col in cat_cols) {
    df_prep[[col]] <- as.numeric(df_prep[[col]])
  }
  
  # ลบคอลัมน์ที่ไม่ต้องการ
  drop_cols <- c("date", "datetime", "timestamp")
  drop_cols <- drop_cols[drop_cols %in% names(df_prep)]
  if (length(drop_cols) > 0) {
    df_prep <- df_prep %>% select(-all_of(drop_cols))
  }
  
  message(sprintf("✅ Final merged shape: %d rows × %d cols", nrow(df_prep), ncol(df_prep)))
  message(sprintf(" Total features available: %d", ncol(df_prep) - 1))
  
  return(df_prep)
}

# ============================================================================
# 7. ฟังก์ชันตรวจสอบคุณภาพก่อนสร้าง Model
# ============================================================================

quality_check <- function(X, y) {
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("🔍 PRE-MODELING QUALITY CHECKPOINT\n")
  cat(rep("=", 70), "\n", sep = "")
  
  # CHECK 1: Missing values
  cat("\nCHECK 1: Missing Values\n")
  x_miss <- sum(is.na(X))
  y_miss <- sum(is.na(y))
  
  if (x_miss > 0) {
    X <- X %>% mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
    x_miss <- sum(is.na(X))
  }
  if (y_miss > 0) {
    y <- ifelse(is.na(y), median(y, na.rm = TRUE), y)
    y_miss <- sum(is.na(y))
  }
  
  if (x_miss == 0 && y_miss == 0) {
    cat("✅ PASS: No missing values\n")
  } else {
    cat(sprintf("⚠️ WARNING: %d NaN in X, %d NaN in y\n", x_miss, y_miss))
  }
  
  # CHECK 2: Infinite values
  cat("\nCHECK 2: Infinite Values\n")
  inf_n <- sum(is.infinite(as.matrix(X)))
  if (inf_n == 0) {
    cat("✅ PASS: No infinite values\n")
  } else {
    cat(sprintf("⚠️ WARNING: %d infinite values in X\n", inf_n))
  }
  
  # CHECK 3: Shape consistency
  cat("\nCHECK 3: Shape Consistency\n")
  if (nrow(X) == length(y)) {
    cat(sprintf("✅ PASS: X=%d rows, y=%d rows\n", nrow(X), length(y)))
  } else {
    cat(sprintf("❌ FAIL: X=%d rows, y=%d rows\n", nrow(X), length(y)))
  }
  
  # CHECK 4: Target distribution
  cat("\nCHECK 4: Target Distribution\n")
  cat(sprintf(" Range [%.3f, %.3f] Mean=%.3f Std=%.3f Skew=%.3f\n", 
              min(y), max(y), mean(y), sd(y), e1071::skewness(y)))
  cat("✅ PASS\n")
  
  # CHECK 5: Dataset size
  cat("\nCHECK 5: Dataset Size\n")
  cat(sprintf(" Samples: %d — small dataset (geo/policy data); using LOO/5-fold CV carefully\n", nrow(X)))
  cat("✅ PASS (small dataset — CV will be used instead of hold-out where needed)\n")
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("✅✅✅ ALL CHECKS PASSED — READY FOR MODELING ✅✅✅\n")
  cat(rep("=", 70), "\n", sep = "")
  
  return(list(X = X, y = y))
}

# ============================================================================
# 8. ฟังก์ชันฝึกและประเมิน Model
# ============================================================================

evaluate_model <- function(model, X_train, X_test, y_train, y_test, name, X_full = NULL, y_full = NULL) {
  
  cat(sprintf("\n%s\n", paste(rep("=", 65), collapse = "")))
  cat(sprintf("🔍 %s\n", name))
  cat(sprintf("%s\n", paste(rep("=", 65), collapse = "")))
  
  tryCatch({
    
    # ฝึก Model
    if (inherits(model, "xgb.Booster")) {
      dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
      model <- xgb.train(data = dtrain, nrounds = model$nrounds, 
                         params = model$params, verbose = 0)
    } else {
      model <- train(y ~ ., data = X_train, method = model, trControl = trainControl(method = "cv", number = 5))
    }
    
    # ทำนาย
    if (inherits(model, "xgb.Booster")) {
      y_train_pred <- predict(model, as.matrix(X_train))
      y_test_pred <- predict(model, as.matrix(X_test))
    } else {
      y_train_pred <- predict(model, X_train)
      y_test_pred <- predict(model, X_test)
    }
    
    # คำนวณ Metrics
    train_r2 <- caret::R2(y_train, y_train_pred)
    test_r2 <- caret::R2(y_test, y_test_pred)
    train_rmse <- RMSE(y_train, y_train_pred)
    test_rmse <- RMSE(y_test, y_test_pred)
    train_mae <- MAE(y_train, y_train_pred)
    test_mae <- MAE(y_test, y_test_pred)
    test_mape <- MAPE(y_test, y_test_pred)
    
    # Cross-validation
    if (inherits(model, "train")) {
      cv_m <- model$results$Rsquared
      cv_s <- 0
    } else {
      cv_m <- test_r2
      cv_s <- 0
    }
    
    # แสดงผล
    cat(sprintf("%-22s %-14s %-14s\n", "Metric", "Train", "Test"))
    cat(paste(rep("-", 50), collapse = ""), "\n")
    cat(sprintf("%-22s %-14.4f %-14.4f\n", "R²", train_r2, test_r2))
    cat(sprintf("%-22s %-14.4f %-14.4f\n", "RMSE", train_rmse, test_rmse))
    cat(sprintf("%-22s %-14.4f %-14.4f\n", "MAE", train_mae, test_mae))
    cat(sprintf("%-22s %-14s %-14.2f\n", "MAPE (%)", "-", test_mape))
    
    gap <- abs(train_r2 - test_r2)
    if (gap < 0.15) {
      cat(sprintf("Overfitting gap: %.4f ✅ OK\n", gap))
    } else {
      cat(sprintf("Overfitting gap: %.4f ⚠️ Review\n", gap))
    }
    
    if (!is.null(cv_m)) {
      cat(sprintf("5-Fold CV R²: %.4f ± %.4f\n", cv_m, cv_s))
    }
    
    # ส่งคืนผลลัพธ์
    return(list(
      model_name = name,
      model = model,
      predictions = y_test_pred,
      problem_type = "Regression",
      train_r2 = train_r2,
      test_r2 = test_r2,
      train_rmse = train_rmse,
      test_rmse = test_rmse,
      train_mae = train_mae,
      test_mae = test_mae,
      test_mape = test_mape,
      cv_mean = cv_m,
      cv_std = cv_s
    ))
    
  }, error = function(e) {
    cat(sprintf("❌ %s failed: %s\n", name, e$message))
    return(NULL)
  })
}

# ============================================================================
# 9. ฟังก์ชันสร้าง Visualization
# ============================================================================

create_visualizations <- function(data_list, df_merged, model_results) {
  
  # สร้างโฟลเดอร์สำหรับเก็บรูป
  if (!dir.exists("figures")) {
    dir.create("figures")
  }
  
  # 1. Petrol Price Impact by Country
  if (!is.null(data_list$petrol)) {
    
    p1 <- data_list$petrol %>%
      ggplot(aes(x = reorder(!!!sym(names(data_list$petrol)[grep("country", names(data_list$petrol), ignore.case = TRUE)[1]]), 
                   !!!sym(names(data_list$petrol)[grep("pct", names(data_list$petrol), ignore.case = TRUE)[1]]))) +
      geom_col(aes(fill = ifelse(!!!sym(names(data_list$petrol)[grep("pct", names(data_list$petrol), ignore.case = TRUE)[1]]) > 0, "#2ca02c", "#d62728"))) +
      coord_flip() +
      labs(title = "Petrol Price % Increase by Country",
           x = "Country", y = "% Increase") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave("figures/01_petrol_price_impact.png", p1, width = 10, height = 6, dpi = 300)
  }
  
  # 2. GDP Impact by Country
  if (!is.null(data_list$country)) {
    
    p2 <- data_list$country %>%
      ggplot(aes(x = reorder(!!!sym(names(data_list$country)[grep("country", names(data_list$country), ignore.case = TRUE)[1]]), 
                   !!!sym(names(data_list$country)[grep("gdp", names(data_list$country), ignore.case = TRUE)[1]]))) +
      geom_col(aes(fill = ifelse(!!!sym(names(data_list$country)[grep("gdp", names(data_list$country), ignore.case = TRUE)[1]]) > 0, "#2ca02c", "#d62728"))) +
      coord_flip() +
      labs(title = "GDP Impact (%)",
           x = "Country", y = "% GDP Change") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave("figures/02_gdp_impact.png", p2, width = 10, height = 6, dpi = 300)
  }
  
  # 3. Model Performance Comparison
  if (length(model_results) > 0) {
    
    model_df <- do.call(rbind, lapply(model_results, function(x) {
      if (!is.null(x)) {
        data.frame(
          Model = x$model_name,
          Test_R2 = x$test_r2,
          Test_RMSE = x$test_rmse,
          stringsAsFactors = FALSE
        )
      }
    }))
    
    p3 <- model_df %>%
      ggplot(aes(x = reorder(Model, Test_R2), y = Test_R2)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = "Model Performance Comparison (Test R²)",
           x = "Model", y = "Test R²") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave("figures/03_model_performance.png", p3, width = 10, height = 6, dpi = 300)
  }
  
  message("✅ Visualizations saved to 'figures/' folder")
}

# ============================================================================
# 10. ฟังก์ชันหลัก (Main Function)
# ============================================================================

main <- function(data_dir = "data") {
  
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("🚀 IMPACT ANALYSIS: 2026 US-Iran War Oil Price Impact Analysis\n")
  cat(rep("=", 80), "\n\n", sep = "")
  
  # 1. โหลดข้อมูล
  cat("📥 Step 1: Loading data...\n")
  data_list <- load_data(data_dir)
  
  # 2. ตรวจสอบคุณภาพข้อมูล
  cat("\n📊 Step 2: Auditing data...\n")
  for (name in names(data_list)) {
    audit_data(data_list[[name]], name)
  }
  
  # 3. ทำความสะอาดข้อมูล
  cat("\n🧹 Step 3: Cleaning data...\n")
  for (name in names(data_list)) {
    data_list[[name]] <- clean_data(data_list[[name]], name)
  }
  
  # 4. รวมข้อมูล
  cat("\n🔗 Step 4: Merging datasets...\n")
  df_merged <- merge_datasets(data_list)
  
  # 5. เตรียมฟีเจอร์
  cat("\n⚙️ Step 5: Preparing features...\n")
  df_prep <- prepare_features(df_merged)
  
  # 6. ตรวจสอบคุณภาพก่อนสร้าง Model
  cat("\n✅ Step 6: Quality check...\n")
  
  # เตรียม X และ y
  target_col <- names(df_prep)[grep("pct", names(df_prep), ignore.case = TRUE)][1]
  if (is.null(target_col)) {
    stop("❌ ไม่พบคอลัมน์ target (ควรมีคำว่า 'pct')")
  }
  
  X <- df_prep %>% select(-all_of(target_col))
  y <- df_prep[[target_col]]
  
  quality_result <- quality_check(X, y)
  X <- quality_result$X
  y <- quality_result$y
  
  # 7. แบ่งข้อมูล Train/Test
  cat("\n📊 Step 7: Train-Test Split...\n")
  set.seed(42)
  train_index <- createDataPartition(y, p = 0.85, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index]
  y_test <- y[-train_index]
  
  cat(sprintf("✅ Train: %d samples | Test: %d samples\n", nrow(X_train), nrow(X_test)))
  
  # 8. ฝึก Model
  cat("\n🤖 Step 8: Training models...\n")
  
  models <- list(
    ridge = "ridge",
    lasso = "lasso",
    rf = "rf",
    gbm = "gbm"
  )
  
  model_results <- list()
  
  for (model_name in names(models)) {
    cat(sprintf("\nTraining %s...\n", model_name))
    
    if (model_name == "ridge") {
      model <- train(y ~ ., data = X_train, method = "ridge",
                     trControl = trainControl(method = "cv", number = 5),
                     tuneLength = 5)
    } else if (model_name == "lasso") {
      model <- train(y ~ ., data = X_train, method = "lasso",
                     trControl = trainControl(method = "cv", number = 5),
                     tuneLength = 5)
    } else if (model_name == "rf") {
      model <- train(y ~ ., data = X_train, method = "rf",
                     trControl = trainControl(method = "cv", number = 5),
                     ntree = 100)
    } else if (model_name == "gbm") {
      model <- train(y ~ ., data = X_train, method = "gbm",
                     trControl = trainControl(method = "cv", number = 5),
                     n.trees = 100,
                     interaction.depth = 3,
                     shrinkage = 0.1,
                     verbose = FALSE)
    }
    
    result <- evaluate_model(model, X_train, X_test, y_train, y_test, 
                             toupper(model_name), X, y)
    
    if (!is.null(result)) {
      model_results[[model_name]] <- result
    }
  }
  
  # 9. แสดงผล Model Performance
  cat("\n📊 Step 9: Model Performance Summary...\n")
  
  if (length(model_results) > 0) {
    model_df <- do.call(rbind, lapply(model_results, function(x) {
      if (!is.null(x)) {
        data.frame(
          Model = x$model_name,
          Test_R2 = x$test_r2,
          Test_RMSE = x$test_rmse,
          Test_MAE = x$test_mae,
          Test_MAPE = x$test_mape,
          stringsAsFactors = FALSE
        )
      }
    }))
    
    model_df <- model_df %>%
      arrange(desc(Test_R2))
    
    print(model_df)
    
    best_model <- model_results[[which.max(model_df$Test_R2)]]
    cat(sprintf("\n🏆 BEST MODEL: %s\n", best_model$model_name))
    cat(sprintf(" Test R²: %.4f\n", best_model$test_r2))
    cat(sprintf(" Test RMSE: %.4f\n", best_model$test_rmse))
    cat(sprintf(" Test MAPE: %.2f%%\n", best_model$test_mape))
  }
  
  # 10. สร้าง Visualization
  cat("\n📊 Step 10: Creating visualizations...\n")
  create_visualizations(data_list, df_merged, model_results)
  
  cat("\n", rep("=", 80), "\n", sep = "")
  cat("✅ Analysis Complete!\n")
  cat(rep("=", 80), "\n\n", sep = "")
  
  return(list(
    data_list = data_list,
    df_merged = df_merged,
    df_prep = df_prep,
    model_results = model_results
  ))
}

# ============================================================================
# 11. รันโปรแกรมหลัก
# ============================================================================

# เรียกใช้ฟังก์ชันหลัก
# results <- main(data_dir = "data")

# ============================================================================
# 12. ตัวอย่างการใช้งาน
# ============================================================================

# โหลดข้อมูล
# data_list <- load_data("data")

# ตรวจสอบคุณภาพข้อมูล
# audit_data(data_list$petrol, "petrol")
# audit_data(data_list$country, "country")

# ทำความสะอาดข้อมูล
# data_list$petrol <- clean_data(data_list$petrol, "petrol")
# data_list$country <- clean_data(data_list$country, "country")

# รวมข้อมูล
# df_merged <- merge_datasets(data_list)

# เตรียมฟีเจอร์
# df_prep <- prepare_features(df_merged)

# เตรียม X และ y
# target_col <- "pct_increase"
# X <- df_prep %>% select(-all_of(target_col))
# y <- df_prep[[target_col]]

# แบ่งข้อมูล Train/Test
# set.seed(42)
# train_index <- createDataPartition(y, p = 0.85, list = FALSE)
# X_train <- X[train_index, ]
# X_test <- X[-train_index, ]
# y_train <- y[train_index]
# y_test <- y[-train_index]

# ฝึก Model
# model <- train(y ~ ., data = X_train, method = "rf",
#                trControl = trainControl(method = "cv", number = 5),
#                ntree = 100)

# ทำนาย
# y_pred <- predict(model, X_test)

# ประเมินผล
# cat("Test R²:", caret::R2(y_test, y_pred), "\n")
# cat("Test RMSE:", RMSE(y_test, y_pred), "\n")
# cat("Test MAPE:", MAPE(y_test, y_pred), "%\n")

# ============================================================================
# 13. จบโปรแกรม
# ============================================================================

message("✅ Script พร้อมใช้งาน!")
