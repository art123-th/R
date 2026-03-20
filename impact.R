---
title: "2026 US-Iran War Oil Price Impact Analysis"
output: html_notebook
---

```{r setup, message=FALSE, warning=FALSE}
# ============================================================================
# IMPACT ANALYSIS: 2026 US-Iran War Oil Price Impact Analysis
# ภาษา: R  |  อ้างอิงจาก Petrol.ipynb
# ============================================================================

# ============================================================================
# 1. โหลด Library ที่จำเป็น
# ============================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(gbm)
library(corrplot)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(scales)
library(gridExtra)
library(e1071)

# Optional packages (ลอง load — ถ้าไม่มีจะข้ามไป)
XGB_OK <- requireNamespace("xgboost",  quietly = TRUE)
LGB_OK <- requireNamespace("lightgbm", quietly = TRUE)

if (XGB_OK) library(xgboost)  else message("⚠️  XGBoost not available — will skip")
if (LGB_OK) library(lightgbm) else message("⚠️  LightGBM not available — will skip")

RANDOM_STATE <- 42
set.seed(RANDOM_STATE)

# ตั้งค่า theme สำหรับกราฟ
theme_set(theme_minimal(base_size = 11))

message("✅ Environment configured")

# ============================================================================
# 2. Smart Loader — โหลดไฟล์ CSV ด้วย keyword matching + positional fallback
# ============================================================================

load_data_smart <- function(data_dir = "data") {

  # สแกนหาไฟล์ทั้งหมด
  all_files <- list.files(data_dir, full.names = TRUE, recursive = TRUE)
  csv_files <- all_files[grepl("\\.csv$", all_files, ignore.case = TRUE)]

  message(sprintf("\n📊 Discovery Summary:"))
  message(sprintf("   Total files : %d", length(all_files)))
  message(sprintf("   CSV files   : %d", length(csv_files)))

  if (length(csv_files) == 0) stop("❌ No CSV files found! Check data directory.")
  message("✅ CSVs discovered — proceeding to load\n")

  # Keyword map (เทียบกับ KEYWORD_MAP ใน Python)
  keyword_map <- list(
    crude     = c("crude"),
    petrol    = c("petrol", "price"),
    country   = c("country", "impact"),
    pros_cons = c("pros",   "cons"),
    timeline  = c("timeline", "war")
  )

  result    <- list()
  used_fps  <- c()

  # Keyword matching
  for (key in names(keyword_map)) {
    keywords <- keyword_map[[key]]
    matched  <- NULL

    for (fp in csv_files) {
      fn <- tolower(basename(fp))
      if (any(sapply(keywords, function(kw) grepl(kw, fn, fixed = TRUE)))) {
        matched <- fp
        break
      }
    }

    if (!is.null(matched)) {
      tryCatch({
        result[[key]] <- read_csv(matched, show_col_types = FALSE)
        used_fps      <- c(used_fps, matched)
        message(sprintf("✅  [%-20s] ← %s  (%d rows × %d cols)",
                        key, basename(matched),
                        nrow(result[[key]]), ncol(result[[key]])))
      }, error = function(e) {
        message(sprintf("❌  [%s] load error: %s", key, e$message))
        result[[key]] <<- NULL
      })
    } else {
      message(sprintf("⚠️  [%-20s] No match found — will try positional fallback", key))
      result[[key]] <- NULL
    }
  }

  # Positional fallback สำหรับ key ที่ยังเป็น NULL
  none_keys  <- names(result)[sapply(result, is.null)]
  unused_fps <- csv_files[!csv_files %in% used_fps]

  for (i in seq_along(none_keys)) {
    k <- none_keys[i]
    if (i <= length(unused_fps)) {
      tryCatch({
        result[[k]] <- read_csv(unused_fps[i], show_col_types = FALSE)
        message(sprintf("✅  [%-20s] ← fallback: %s", k, basename(unused_fps[i])))
      }, error = function(e) {
        message(sprintf("❌  [%s] fallback failed: %s", k, e$message))
      })
    }
  }

  message("\n✅ All files loaded!")
  return(result)
}

# ============================================================================
# 3. ฟังก์ชันตรวจสอบคุณภาพข้อมูล (Data Audit)
# ============================================================================

audit_data <- function(df, name) {

  if (is.null(df)) {
    message(sprintf("⚠️  %s: DataFrame is None — skipping audit\n", name))
    return(invisible(NULL))
  }

  cat(rep("=", 70), "\n", sep = "")
  cat(sprintf("📊 AUDIT: %s\n", name))
  cat(rep("=", 70), "\n", sep = "")
  cat(sprintf("  Shape   : %s rows × %d cols\n", format(nrow(df), big.mark = ","), ncol(df)))
  cat(sprintf("  Memory  : %.1f KB\n", as.numeric(object.size(df)) / 1024))

  cat("\n  Columns:\n")
  for (i in seq_along(names(df))) {
    col_name <- names(df)[i]
    col_type <- class(df[[col_name]])[1]
    n_unique <- n_distinct(df[[col_name]], na.rm = TRUE)
    n_miss   <- sum(is.na(df[[col_name]]))
    cat(sprintf("  %2d. %-35s dtype=%-12s unique=%-6d missing=%d\n",
                i, sprintf("'%s'", col_name), col_type, n_unique, n_miss))
  }

  dupes     <- sum(duplicated(df))
  total_na  <- sum(is.na(df))
  cat(sprintf("\n  Duplicates: %d\n", dupes))
  cat(sprintf("  Total NaN : %d\n\n", total_na))

  return(invisible(NULL))
}

# ============================================================================
# 4. ฟังก์ชันทำความสะอาดข้อมูล (Comprehensive Cleaning — ตรงกับ clean_df Python)
# ============================================================================

clean_df <- function(df, name) {

  if (is.null(df)) {
    message(sprintf("⚠️  %s: None — skipped", name)); return(NULL)
  }

  d <- df
  cat(rep("─", 60), "\n", sep = "")
  cat(sprintf("🧹 Cleaning: %s  |  Start: %d×%d  |  NaN: %d\n",
              name, nrow(d), ncol(d), sum(is.na(d))))

  # Parse คอลัมน์วันที่
  date_cols <- names(d)[grepl("date", names(d), ignore.case = TRUE)]
  for (col in date_cols) {
    parsed <- tryCatch(
      parse_date_time(as.character(d[[col]]), orders = c("ymd", "dmy", "mdy", "ymd HMS"), quiet = TRUE),
      error = function(e) NULL
    )
    if (!is.null(parsed) && sum(!is.na(parsed)) > 0) {
      d[[col]] <- parsed
      cat(sprintf("  ✅ Parsed '%s' as datetime\n", col))
    }
  }

  # Numeric imputation ด้วย median (ไม่ drop แบบ na.omit)
  num_cols <- names(d)[sapply(d, is.numeric)]
  for (col in num_cols) {
    n_miss <- sum(is.na(d[[col]]))
    if (n_miss > 0) {
      fv <- median(d[[col]], na.rm = TRUE)
      # ป้องกัน numeric(0) เมื่อทุก value เป็น NA
      if (length(fv) == 0 || is.na(fv)) {
        d[[col]][is.na(d[[col]])] <- 0
        cat(sprintf("  ⚠️  '%s': all NA — filled %d NaN with 0\n", col, n_miss))
      } else {
        d[[col]][is.na(d[[col]])] <- fv
        cat(sprintf("  ✅ '%s': filled %d NaN with median=%.3f\n", col, n_miss, fv))
      }
    }
  }

  # Categorical imputation ด้วย mode
  char_cols <- names(d)[sapply(d, function(x) is.character(x) | is.factor(x))]
  for (col in char_cols) {
    n_miss <- sum(is.na(d[[col]]))
    if (n_miss > 0) {
      modes <- names(sort(table(d[[col]]), decreasing = TRUE))
      fv    <- if (length(modes) > 0) modes[1] else "Unknown"
      d[[col]][is.na(d[[col]])] <- fv
      cat(sprintf("  ✅ '%s': filled %d NaN with mode='%s'\n", col, n_miss, fv))
    }
  }

  # ลบคอลัมน์ที่ missing > 50%
  miss_pct  <- colSums(is.na(d)) / nrow(d)
  drop_cols <- names(miss_pct[miss_pct > 0.5])
  if (length(drop_cols) > 0) {
    d <- d %>% select(-all_of(drop_cols))
    cat(sprintf("  ⚠️  Dropped %d high-missing cols: %s\n",
                length(drop_cols), paste(drop_cols, collapse = ", ")))
  }

  # ลบแถวที่ยัง NA (ถ้า < 30%)
  rem <- sum(is.na(d))
  if (rem > 0) {
    rows_with <- sum(apply(d, 1, anyNA))
    pct       <- 100 * rows_with / nrow(d)
    if (pct < 30) {
      d <- d %>% drop_na()
      cat(sprintf("  ✅ Dropped %d rows with remaining NaN\n", rows_with))
    } else {
      cat(sprintf("  ❌ %.1f%% rows affected — manual review needed\n", pct))
    }
  }

  # ลบแถวซ้ำ
  dup <- sum(duplicated(d))
  if (dup > 0) {
    d <- distinct(d)
    cat(sprintf("  ✅ Removed %d duplicates\n", dup))
  }

  final_miss <- sum(is.na(d))
  stopifnot("❌ FATAL: NaN remain after cleaning!" = final_miss == 0)
  cat(sprintf("  ✅ CLEAN  |  End: %d×%d  |  NaN: %d\n", nrow(d), ncol(d), final_miss))

  return(d)
}

# ============================================================================
# 5. Helper functions (เทียบกับ safe_divide / safe_log ใน Python)
# ============================================================================

safe_divide <- function(num, denom, fill = 0) {
  r <- num / (denom + 1e-10)
  r[is.infinite(r) | is.nan(r)] <- fill
  r[is.na(r)]                   <- fill
  return(r)
}

safe_log <- function(vals, fill = 0) {
  r <- log1p(pmax(vals, 0))
  r[is.infinite(r) | is.nan(r)] <- fill
  r[is.na(r)]                   <- fill
  return(r)
}

# ============================================================================
# 6. ฟังก์ชันรวมข้อมูลและ Feature Engineering
# ============================================================================

build_features <- function(df_petrol_c, df_country_c) {

  # ── Merge ──────────────────────────────────────────────────────────────
  ctry_p <- names(df_petrol_c)[grep("country", names(df_petrol_c), ignore.case = TRUE)][1]
  ctry_c <- names(df_country_c)[grep("country", names(df_country_c), ignore.case = TRUE)][1]

  if (!is.na(ctry_p) && !is.na(ctry_c)) {
    df_merged <- df_petrol_c %>%
      left_join(df_country_c, by = setNames(ctry_c, ctry_p),
                suffix = c("_petrol", "_country"))
    message(sprintf("✅ Merged shape: %d×%d", nrow(df_merged), ncol(df_merged)))
  } else {
    df_merged <- df_petrol_c
    message("⚠️  Could not merge — using petrol table only")
  }

  # ── Label Encode categoricals ──────────────────────────────────────────
  le_map <- list()
  chr_cols <- names(df_merged)[sapply(df_merged, function(x) is.character(x) | is.factor(x))]
  for (col in chr_cols) {
    lvls            <- sort(unique(as.character(df_merged[[col]])))
    le_map[[col]]   <- lvls
    df_merged[[col]] <- as.integer(factor(df_merged[[col]], levels = lvls))
    message(sprintf("  ✅ LabelEncoded: %s", col))
  }

  # ── Drop datetime columns ──────────────────────────────────────────────
  dt_cols <- names(df_merged)[sapply(df_merged, function(x) inherits(x, c("Date", "POSIXct", "POSIXlt")))]
  if (length(dt_cols) > 0) {
    df_merged <- df_merged %>% select(-all_of(dt_cols))
    message(sprintf("  ✅ Dropped datetime: %s", paste(dt_cols, collapse = ", ")))
  }

  # ── Identify TARGET ────────────────────────────────────────────────────
  target_candidates <- c("Pct_Increase", "pct_increase", "PctIncrease")
  TARGET <- NULL
  for (cand in target_candidates) {
    if (cand %in% names(df_merged)) { TARGET <- cand; break }
  }
  if (is.null(TARGET)) {
    pct_cands <- names(df_merged)[grepl("pct", names(df_merged), ignore.case = TRUE)]
    TARGET    <- if (length(pct_cands) > 0) pct_cands[1] else names(df_merged)[ncol(df_merged)]
  }
  message(sprintf("\n🎯 Target column: '%s'", TARGET))

  # ── Engineered features ────────────────────────────────────────────────
  before_usd_col <- names(df_merged)[grepl("before", names(df_merged), ignore.case = TRUE) &
                                       grepl("usd",    names(df_merged), ignore.case = TRUE)][1]
  after_usd_col  <- names(df_merged)[(grepl("mar7",  names(df_merged), ignore.case = TRUE) |
                                        grepl("after", names(df_merged), ignore.case = TRUE)) &
                                       grepl("usd", names(df_merged), ignore.case = TRUE)][1]
  gdp_col    <- names(df_merged)[grepl("gdp",    names(df_merged), ignore.case = TRUE)][1]
  import_col <- names(df_merged)[grepl("import", names(df_merged), ignore.case = TRUE)][1]
  stock_col  <- names(df_merged)[grepl("stock",  names(df_merged), ignore.case = TRUE)][1]

  if (!is.na(before_usd_col) && !is.na(after_usd_col)) {
    df_merged$price_multiplier <- safe_divide(df_merged[[after_usd_col]],
                                              df_merged[[before_usd_col]])
    message("  ✅ Engineered: price_multiplier")
  }
  if (!is.na(before_usd_col)) {
    df_merged$log_before_usd <- safe_log(df_merged[[before_usd_col]])
    message("  ✅ Engineered: log_before_usd")
  }
  if (!is.na(gdp_col) && !is.na(import_col)) {
    df_merged$gdp_import_interact <- df_merged[[gdp_col]] * df_merged[[import_col]]
    message("  ✅ Engineered: gdp_import_interact")
  }
  if (!is.na(stock_col) && !is.na(import_col)) {
    df_merged$stock_import_ratio <- safe_divide(df_merged[[stock_col]],
                                                df_merged[[import_col]])
    message("  ✅ Engineered: stock_import_ratio")
  }

  message(sprintf("✅ Final merged shape: %d×%d", nrow(df_merged), ncol(df_merged)))
  message(sprintf("   Total features available: %d", ncol(df_merged) - 1))

  return(list(df = df_merged, target = TARGET, le_map = le_map))
}

# ============================================================================
# 7. ฟังก์ชันเลือก Features (พร้อม Leakage exclusion)
# ============================================================================

select_features <- function(df_merged, TARGET) {

  # Leakage keywords (เทียบกับ LEAKAGE_KEYWORDS ใน Python)
  leakage_kw <- c("amount_change", "pct_increase", tolower(TARGET),
                  "mar7_price", "after_price")

  feature_cols <- names(df_merged)[sapply(names(df_merged), function(col) {
    col_lower <- tolower(col)
    is.numeric(df_merged[[col]]) &&
      col != TARGET &&
      !any(sapply(leakage_kw, function(kw) grepl(kw, col_lower, fixed = TRUE)))
  })]

  message(sprintf("\nFeature columns (%d):", length(feature_cols)))
  for (i in seq_along(feature_cols)) {
    message(sprintf("  %2d. %s", i, feature_cols[i]))
  }

  return(feature_cols)
}

# ============================================================================
# 8. ฟังก์ชันตรวจสอบคุณภาพก่อน Modeling (Pre-Modeling Quality Checkpoint)
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
    X       <- X %>% mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
    x_miss  <- sum(is.na(X))
  }
  if (y_miss > 0) {
    y[is.na(y)] <- median(y, na.rm = TRUE)
    y_miss       <- sum(is.na(y))
  }
  stopifnot("❌ NaN remain in X" = x_miss == 0, "❌ NaN remain in y" = y_miss == 0)
  cat("✅ PASS: No missing values\n")

  # CHECK 2: Infinite values
  cat("\nCHECK 2: Infinite Values\n")
  inf_n <- sum(is.infinite(as.matrix(X)))
  if (inf_n > 0) {
    X[sapply(X, is.infinite)] <- NA
    X <- X %>% mutate(across(everything(), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
    inf_n <- sum(is.infinite(as.matrix(X)))
  }
  stopifnot("❌ Infinite values remain in X" = inf_n == 0)
  cat("✅ PASS: No infinite values\n")

  # CHECK 3: All numeric
  cat("\nCHECK 3: Data Types\n")
  non_num <- names(X)[!sapply(X, is.numeric)]
  if (length(non_num) > 0) {
    for (col in non_num) X[[col]] <- as.numeric(as.character(X[[col]]))
    cat(sprintf("  ✅ Converted %d cols\n", length(non_num)))
  } else {
    cat("✅ PASS: All numeric\n")
  }

  # CHECK 4: Shape consistency
  cat("\nCHECK 4: Shape Consistency\n")
  stopifnot("❌ X and y length mismatch!" = nrow(X) == length(y))
  cat(sprintf("✅ PASS: X=%d rows, y=%d rows\n", nrow(X), length(y)))

  # CHECK 5: Target distribution
  cat("\nCHECK 5: Target Distribution\n")
  cat(sprintf("  Range [%.3f, %.3f]  Mean=%.3f  Std=%.3f  Skew=%.3f\n",
              min(y), max(y), mean(y), sd(y), e1071::skewness(y)))
  cat("✅ PASS\n")

  # CHECK 6: Dataset size
  cat("\nCHECK 6: Dataset Size\n")
  cat(sprintf("  Samples: %d — small dataset (geo/policy data); using LOO/5-fold CV carefully\n", nrow(X)))
  cat("✅ PASS (small dataset — CV will be used instead of hold-out where needed)\n")

  cat("\n", rep("=", 70), "\n", sep = "")
  cat("✅✅✅ ALL CHECKS PASSED — READY FOR MODELING ✅✅✅\n")
  cat(rep("=", 70), "\n\n", sep = "")

  return(list(X = X, y = y))
}

# ============================================================================
# 9. ฟังก์ชัน Scaling
# ============================================================================

scale_features <- function(X, X_train, X_test) {
  # StandardScaler (mean=0, sd=1) — ใช้กับ Ridge/Lasso
  col_means <- colMeans(X_train)
  col_sds   <- apply(X_train, 2, sd)
  col_sds[col_sds == 0] <- 1  # หลีกเลี่ยง division by zero

  scale_fn <- function(mat) {
    sweep(sweep(mat, 2, col_means, "-"), 2, col_sds, "/")
  }

  X_sc        <- as.data.frame(scale_fn(as.matrix(X)))
  X_train_sc  <- as.data.frame(scale_fn(as.matrix(X_train)))
  X_test_sc   <- as.data.frame(scale_fn(as.matrix(X_test)))
  names(X_sc) <- names(X_train_sc) <- names(X_test_sc) <- names(X)

  message(sprintf("✅ Scaling complete: X_train_sc=%d×%d, X_test_sc=%d×%d",
                  nrow(X_train_sc), ncol(X_train_sc),
                  nrow(X_test_sc),  ncol(X_test_sc)))

  return(list(X_sc = X_sc, X_train_sc = X_train_sc, X_test_sc = X_test_sc))
}

# ============================================================================
# 10. ฟังก์ชันประเมิน Model (ตรงกับ evaluate_model ใน Python)
# ============================================================================

evaluate_model <- function(model_obj, X_tr, X_te, y_tr, y_te, name,
                           n_cv = NULL) {

  cat(sprintf("\n%s\n", paste(rep("=", 65), collapse = "")))
  cat(sprintf("🔍 %s\n", name))
  cat(sprintf("%s\n", paste(rep("=", 65), collapse = "")))

  tryCatch({

    # Fit
    if (inherits(model_obj, "xgb.Booster")) {
      dtrain  <- xgb.DMatrix(data = as.matrix(X_tr), label = y_tr)
      model_obj <- xgb.train(data = dtrain, params = model_obj$params,
                             nrounds = model_obj$nrounds, verbose = 0)
    } else {
      model_obj <- fit_model(model_obj, X_tr, y_tr)
    }

    # Predict
    y_tr_p <- predict_model(model_obj, X_tr)
    y_te_p <- predict_model(model_obj, X_te)

    # Metrics
    tr_r2   <- cor(y_tr, y_tr_p)^2
    te_r2   <- cor(y_te, y_te_p)^2
    tr_rmse <- sqrt(mean((y_tr - y_tr_p)^2))
    te_rmse <- sqrt(mean((y_te - y_te_p)^2))
    tr_mae  <- mean(abs(y_tr - y_tr_p))
    te_mae  <- mean(abs(y_te - y_te_p))
    te_mape <- mean(abs((y_te - y_te_p) / (abs(y_te) + 1e-10))) * 100

    # Display
    cat(sprintf("%-22s %-14s %-14s\n", "Metric", "Train", "Test"))
    cat(paste(rep("-", 50), collapse = ""), "\n")
    cat(sprintf("%-22s %-14.4f %-14.4f\n", "R²",        tr_r2,   te_r2))
    cat(sprintf("%-22s %-14.4f %-14.4f\n", "RMSE",      tr_rmse, te_rmse))
    cat(sprintf("%-22s %-14.4f %-14.4f\n", "MAE",       tr_mae,  te_mae))
    cat(sprintf("%-22s %-14s %-14.2f\n",   "MAPE (%)",  "-",     te_mape))

    gap <- abs(tr_r2 - te_r2)
    cat(sprintf("\nOverfitting gap: %.4f  %s\n",
                gap, if (gap < 0.15) "✅ OK" else "⚠️ Review"))

    # Cross-validation
    n_cv  <- if (is.null(n_cv)) min(5, nrow(X_tr)) else min(n_cv, nrow(X_tr))
    cv_m  <- cv_s <- NULL
    tryCatch({
      folds  <- createFolds(y_tr, k = n_cv)
      cv_r2s <- sapply(folds, function(idx) {
        m_cv <- fit_model(model_obj, X_tr[-idx, , drop = FALSE], y_tr[-idx])
        p_cv <- predict_model(m_cv, X_tr[idx,  , drop = FALSE])
        if (length(unique(y_tr[idx])) > 1) cor(y_tr[idx], p_cv)^2 else NA
      })
      cv_m <- mean(cv_r2s, na.rm = TRUE)
      cv_s <- sd(cv_r2s,   na.rm = TRUE)
      cat(sprintf("%d-Fold CV R²: %.4f ± %.4f\n", n_cv, cv_m, cv_s))
    }, error = function(e) {
      cat(sprintf("⚠️  CV failed: %s\n", e$message))
    })

    return(list(
      model_name = name,  model      = model_obj,  predictions = y_te_p,
      problem_type = "Regression",
      train_r2  = tr_r2,   test_r2   = te_r2,
      train_rmse= tr_rmse, test_rmse = te_rmse,
      train_mae = tr_mae,  test_mae  = te_mae,
      test_mape = te_mape, cv_mean   = cv_m, cv_std = cv_s
    ))

  }, error = function(e) {
    cat(sprintf("❌ %s failed: %s\n", name, e$message))
    return(NULL)
  })
}

# Helper: fit model ตามประเภท
fit_model <- function(model_obj, X_tr, y_tr) {
  if (inherits(model_obj, c("randomForest", "RandomForest"))) {
    df_tr <- cbind(X_tr, .y = y_tr)
    randomForest(.y ~ ., data = df_tr,
                 ntree      = model_obj$ntree %||% 200,
                 max_depth  = 5)
  } else if (inherits(model_obj, "gbm")) {
    gbm_model <- gbm::gbm(y_tr ~ ., data = cbind(X_tr, y_tr = y_tr),
                          distribution = "gaussian",
                          n.trees = 100, interaction.depth = 3,
                          shrinkage = 0.1, verbose = FALSE)
    gbm_model
  } else {
    # glmnet-based Ridge / Lasso
    model_obj$fit(X_tr, y_tr)
    model_obj
  }
}

# Helper: predict
predict_model <- function(model_obj, X_te) {
  if (inherits(model_obj, c("randomForest", "RandomForest"))) {
    predict(model_obj, X_te)
  } else if (inherits(model_obj, "gbm")) {
    gbm::predict.gbm(model_obj, X_te, n.trees = 100)
  } else if (inherits(model_obj, "xgb.Booster")) {
    predict(model_obj, xgb.DMatrix(as.matrix(X_te)))
  } else if (inherits(model_obj, "lgb.Booster")) {
    predict(model_obj, as.matrix(X_te))
  } else {
    predict(model_obj, X_te)
  }
}

# Null-coalescing operator
`%||%` <- function(a, b) if (!is.null(a)) a else b

# ============================================================================
# 11. ฟังก์ชันสร้าง Visualization (ตรงกับทุก Cell ใน Petrol.ipynb)
# ============================================================================

# --- 11a. Crude Oil Charts (Cell 5) ---
plot_crude_oil <- function(df_crude_c) {

  if (is.null(df_crude_c)) {
    message("⚠️  Crude oil DataFrame not available"); return(invisible(NULL))
  }

  dc        <- df_crude_c
  num_cols  <- names(dc)[sapply(dc, is.numeric)]
  date_col  <- names(dc)[grep("date",  names(dc), ignore.case = TRUE)][1]
  brent_col <- intersect(
    grep("brent", num_cols, ignore.case = TRUE, value = TRUE),
    grep("usd",   num_cols, ignore.case = TRUE, value = TRUE)
  )[1]
  wti_col   <- intersect(
    grep("wti",   num_cols, ignore.case = TRUE, value = TRUE),
    grep("usd",   num_cols, ignore.case = TRUE, value = TRUE)
  )[1]
  phase_col <- names(dc)[grep("phase", names(dc), ignore.case = TRUE)][1]

  if (!is.na(date_col)) dc <- dc %>% arrange(!!sym(date_col))
  dc$.idx <- seq_len(nrow(dc))

  if (is.na(brent_col) || is.na(wti_col)) {
    message("⚠️  Brent/WTI columns not found — skipping crude plot"); return(invisible(NULL))
  }

  # Plot 1: Brent & WTI prices
  p1 <- ggplot(dc, aes(x = .idx)) +
    geom_line(aes(y = !!sym(brent_col), colour = "Brent USD"), size = 1) +
    geom_point(aes(y = !!sym(brent_col), colour = "Brent USD"), size = 2) +
    geom_line(aes(y = !!sym(wti_col),   colour = "WTI USD"),   size = 1) +
    geom_point(aes(y = !!sym(wti_col),  colour = "WTI USD"),   size = 2) +
    scale_colour_manual(values = c("Brent USD" = "steelblue", "WTI USD" = "darkorange")) +
    labs(title = "Brent & WTI Crude Prices (USD/barrel)",
         x = NULL, y = "USD / barrel", colour = NULL) +
    theme_minimal()

  # Plot 2: % Daily Change
  brent_chg <- intersect(
    grep("brent", num_cols, ignore.case = TRUE, value = TRUE),
    grep("pct",   num_cols, ignore.case = TRUE, value = TRUE)
  )[1]
  wti_chg   <- intersect(
    grep("wti",   num_cols, ignore.case = TRUE, value = TRUE),
    grep("pct",   num_cols, ignore.case = TRUE, value = TRUE)
  )[1]

  p2 <- if (!is.na(brent_chg) && !is.na(wti_chg)) {
    ggplot(dc, aes(x = .idx)) +
      geom_col(aes(y = !!sym(brent_chg)), fill = "steelblue",  alpha = 0.7) +
      geom_col(aes(y = !!sym(wti_chg)),   fill = "darkorange", alpha = 0.5) +
      geom_hline(yintercept = 0, colour = "black", size = 0.8) +
      labs(title = "Daily % Price Change", x = NULL, y = "% Change") +
      theme_minimal()
  } else {
    ggplot() + labs(title = "Daily % Change (no data)") + theme_void()
  }

  # Plot 3: Cumulative returns
  dc$.cum_brent <- (dc[[brent_col]] / dc[[brent_col]][1] - 1) * 100
  dc$.cum_wti   <- (dc[[wti_col]]   / dc[[wti_col]][1]   - 1) * 100

  p3 <- ggplot(dc, aes(x = .idx)) +
    geom_ribbon(aes(ymin = 0, ymax = .cum_brent), fill = "steelblue",  alpha = 0.4) +
    geom_ribbon(aes(ymin = 0, ymax = .cum_wti),   fill = "darkorange", alpha = 0.4) +
    geom_line(aes(y = .cum_brent), colour = "steelblue",  size = 1) +
    geom_line(aes(y = .cum_wti),   colour = "darkorange", size = 1) +
    labs(title = "Cumulative Price Change Since Pre-Conflict",
         x = NULL, y = "Cumulative % Change") +
    theme_minimal()

  # Plot 4: Brent vs WTI scatter
  p4 <- ggplot(dc, aes(x = !!sym(brent_col), y = !!sym(wti_col))) +
    geom_point(colour = "steelblue", size = 3, alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE, colour = "red", linetype = "dashed") +
    labs(title = "Brent vs WTI Scatter",
         x = "Brent USD/barrel", y = "WTI USD/barrel") +
    theme_minimal()

  combined <- grid.arrange(p1, p2, p3, p4, ncol = 2,
                           top = "🛢️ Crude Oil — 2026 US-Iran War Price Surge")
  ggsave("figures/01_crude_oil.png", combined, width = 16, height = 10, dpi = 120)
  message("✅ Saved: figures/01_crude_oil.png")
}

# --- 11b. Petrol Price Charts (Cell 6) ---
plot_petrol_prices <- function(df_petrol_c) {

  if (is.null(df_petrol_c)) {
    message("⚠️  Petrol DataFrame not available"); return(invisible(NULL))
  }

  dp <- df_petrol_c
  message("Columns: ", paste(names(dp), collapse = ", "))

  country_col <- names(dp)[grep("country",  names(dp), ignore.case = TRUE)][1]
  pct_col     <- names(dp)[grep("pct",      names(dp), ignore.case = TRUE)][1]
  before_usd  <- names(dp)[grep("before",   names(dp), ignore.case = TRUE) &
                              grep("usd",   names(dp), ignore.case = TRUE)][1]
  after_usd   <- names(dp)[(grepl("mar7",  names(dp), ignore.case = TRUE) |
                               grepl("after", names(dp), ignore.case = TRUE)) &
                              grepl("usd",   names(dp), ignore.case = TRUE)][1]
  region_col  <- names(dp)[grep("region",   names(dp), ignore.case = TRUE)][1]
  import_dep  <- names(dp)[grep("import|dep", names(dp), ignore.case = TRUE)][1]

  message(sprintf("Detected → country:%s, pct:%s, before_usd:%s, after_usd:%s",
                  country_col, pct_col, before_usd, after_usd))

  plot_list <- list()

  # Plot 1: % Increase horizontal bar
  if (!is.na(country_col) && !is.na(pct_col)) {
    dp_sorted <- dp %>% arrange(!!sym(pct_col))
    dp_sorted$.colour <- ifelse(dp_sorted[[pct_col]] > 0, "#d62728", "#2ca02c")

    plot_list[[1]] <- ggplot(dp_sorted,
                             aes(x = reorder(!!sym(country_col), !!sym(pct_col)),
                                 y = !!sym(pct_col), fill = .colour)) +
      geom_col(colour = "black", size = 0.3) +
      geom_text(aes(label = sprintf("%.1f%%", !!sym(pct_col))),
                hjust = -0.1, size = 3) +
      scale_fill_identity() +
      coord_flip() +
      geom_hline(yintercept = 0, colour = "black", size = 0.6) +
      labs(title = "Petrol Price % Increase per Country",
           x = NULL, y = "% Price Increase") +
      theme_minimal()
  }

  # Plot 2: Before vs After USD
  if (!is.na(country_col) && !is.na(before_usd) && !is.na(after_usd)) {
    # สร้าง long format โดยไม่ใช้ pivot_longer เพื่อหลีกเลี่ยง type mismatch
    dp_long <- rbind(
      data.frame(
        country = dp[[country_col]],
        price   = as.numeric(dp[[before_usd]]),
        period  = "Before War",
        stringsAsFactors = FALSE
      ),
      data.frame(
        country = dp[[country_col]],
        price   = as.numeric(dp[[after_usd]]),
        period  = "After (Mar7)",
        stringsAsFactors = FALSE
      )
    )
    dp_long$period <- factor(dp_long$period,
                             levels = c("Before War", "After (Mar7)"))

    plot_list[[2]] <- ggplot(dp_long,
                             aes(x = country, y = price, fill = period)) +
      geom_col(position = position_dodge(0.8), width = 0.7,
               colour = "black", size = 0.3) +
      scale_fill_manual(values = c("Before War" = "steelblue", "After (Mar7)" = "coral")) +
      labs(title = "Before vs After Price (USD/L)",
           x = NULL, y = "USD / litre", fill = NULL) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))
  }

  # Plot 3: % Increase by Region
  if (!is.na(region_col) && !is.na(pct_col)) {
    reg_grp <- dp %>%
      group_by(!!sym(region_col)) %>%
      summarise(mean_pct = mean(!!sym(pct_col), na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(mean_pct))

    plot_list[[3]] <- ggplot(reg_grp,
                             aes(x = reorder(!!sym(region_col), mean_pct), y = mean_pct)) +
      geom_col(fill = "steelblue", colour = "black", size = 0.3) +
      geom_text(aes(label = sprintf("%.1f%%", mean_pct)), vjust = -0.3, size = 3.5) +
      labs(title = "Avg Price Increase by Region",
           x = NULL, y = "Mean % Increase") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
  }

  # Plot 4: Box by import dependency
  if (!is.na(import_dep) && !is.na(pct_col)) {
    imp_vals <- suppressWarnings(as.numeric(as.character(dp[[import_dep]])))
    if (sum(!is.na(imp_vals)) >= 3) {
      dp$.imp_grp <- cut(imp_vals, breaks = 3,
                         labels = c("Low Import", "Mid Import", "High Import"))
    } else {
      dp$.imp_grp <- factor(rep("Low Import", nrow(dp)))
    }
    plot_list[[4]] <- ggplot(dp, aes(x = .imp_grp, y = !!sym(pct_col), fill = .imp_grp)) +
      geom_boxplot(outlier.colour = "red", alpha = 0.7) +
      labs(title = "% Increase by Import Dependency",
           x = "Import Dependency", y = "% Increase") +
      theme_minimal() + theme(legend.position = "none")
  }

  # Assemble
  n <- length(plot_list)
  if (n > 0) {
    combined <- do.call(grid.arrange, c(plot_list, ncol = 2,
                        list(top = "⛽ Global Petrol Prices — Before vs After US-Iran War")))
    ggsave("figures/02_petrol_prices.png", combined, width = 16, height = 11, dpi = 120)
    message("✅ Saved: figures/02_petrol_prices.png")
  }
}

# --- 11c. Country Economic Impact Charts (Cell 7) ---
plot_country_impact <- function(df_country_c) {

  if (is.null(df_country_c)) {
    message("⚠️  Country DataFrame not available"); return(invisible(NULL))
  }

  dc         <- df_country_c
  gdp_col    <- names(dc)[grep("gdp",      names(dc), ignore.case = TRUE)][1]
  stock_col  <- names(dc)[grep("stock",    names(dc), ignore.case = TRUE)][1]
  import_col <- names(dc)[grep("import",   names(dc), ignore.case = TRUE)][1]
  ctry_col   <- names(dc)[grep("country",  names(dc), ignore.case = TRUE)][1]
  vuln_col   <- names(dc)[grep("vulnerab", names(dc), ignore.case = TRUE)][1]
  region_col <- names(dc)[grep("region",   names(dc), ignore.case = TRUE)][1]

  plot_list <- list()

  # GDP Impact
  if (!is.na(gdp_col) && !is.na(ctry_col)) {
    ds <- dc %>% arrange(!!sym(gdp_col)) %>%
      mutate(.colour = ifelse(!!sym(gdp_col) < 0, "#d62728", "#2ca02c"))
    plot_list[[1]] <- ggplot(ds, aes(x = reorder(!!sym(ctry_col), !!sym(gdp_col)),
                                     y = !!sym(gdp_col), fill = .colour)) +
      geom_col(colour = "black", size = 0.3) +
      geom_text(aes(label = sprintf("%.1f%%", !!sym(gdp_col))),
                hjust = ifelse(ds[[gdp_col]] >= 0, -0.1, 1.1), size = 2.8) +
      scale_fill_identity() +
      geom_vline(xintercept = 0, colour = "black") +
      coord_flip() +
      labs(title = "GDP Impact by Country", x = NULL, y = "GDP Impact (%)") +
      theme_minimal()
  }

  # Stock Market Impact
  if (!is.na(stock_col) && !is.na(ctry_col)) {
    ds2 <- dc %>% arrange(!!sym(stock_col)) %>%
      mutate(.colour2 = ifelse(!!sym(stock_col) < 0, "#d62728", "#2ca02c"))
    plot_list[[2]] <- ggplot(ds2, aes(x = reorder(!!sym(ctry_col), !!sym(stock_col)),
                                      y = !!sym(stock_col), fill = .colour2)) +
      geom_col(colour = "black", size = 0.3) +
      scale_fill_identity() +
      geom_vline(xintercept = 0, colour = "black") +
      coord_flip() +
      labs(title = "Stock Market Impact", x = NULL, y = "Stock Market Change (%)") +
      theme_minimal()
  }

  # Import vs GDP scatter
  if (!is.na(import_col) && !is.na(gdp_col)) {
    vuln_colours <- c(Critical = "#d62728", High = "#ff7f0e",
                      Moderate = "#2ca02c", Low = "#1f77b4")

    p_scatter <- ggplot(dc, aes(x = !!sym(import_col), y = !!sym(gdp_col)))

    if (!is.na(vuln_col)) {
      p_scatter <- p_scatter +
        geom_point(aes(colour = !!sym(vuln_col)), size = 3) +
        scale_colour_manual(values = vuln_colours, name = "Vulnerability")
    } else {
      p_scatter <- p_scatter +
        geom_point(colour = "steelblue", size = 3)
    }

    if (!is.na(ctry_col)) {
      p_scatter <- p_scatter +
        geom_text(aes(label = !!sym(ctry_col)),
                  size = 2.5, hjust = 0, vjust = 0, alpha = 0.7)
    }

    plot_list[[3]] <- p_scatter +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "black") +
      labs(title = "Oil Import % vs GDP Impact",
           x = "Oil Import Dependency (%)", y = "GDP Impact (%)") +
      theme_minimal()
  }

  # Vulnerability distribution
  if (!is.na(vuln_col)) {
    vuln_order <- c("Critical", "High", "Moderate", "Low")
    dc$.vuln_f  <- factor(dc[[vuln_col]], levels = vuln_order)
    plot_list[[4]] <- ggplot(dc %>% count(.vuln_f),
                             aes(x = .vuln_f, y = n, fill = .vuln_f)) +
      geom_col(colour = "black", size = 0.3) +
      scale_fill_manual(values = c(Critical = "#d62728", High = "#ff7f0e",
                                   Moderate = "#ffbb78", Low = "#2ca02c"),
                        na.value = "grey60") +
      geom_text(aes(label = n), vjust = -0.3, fontface = "bold") +
      labs(title = "Vulnerability Distribution", x = NULL, y = "Count") +
      theme_minimal() + theme(legend.position = "none")
  }

  n <- length(plot_list)
  if (n > 0) {
    combined <- do.call(grid.arrange, c(plot_list, ncol = 2,
                        list(top = "🌍 Country Economic Impact — 2026 Oil Shock")))
    ggsave("figures/03_country_impact.png", combined, width = 16, height = 11, dpi = 120)
    message("✅ Saved: figures/03_country_impact.png")
  }
}

# --- 11d. Timeline Charts (Cell 8) ---
plot_timeline <- function(df_timeline_c) {

  if (is.null(df_timeline_c)) {
    message("⚠️  Timeline DataFrame not available"); return(invisible(NULL))
  }

  dt       <- df_timeline_c
  cat_col  <- names(dt)[grep("categ", names(dt), ignore.case = TRUE)][1]
  date_col <- names(dt)[grep("date",  names(dt), ignore.case = TRUE)][1]

  plot_list <- list()

  # Events by Category (pie → bar ใน R เพื่อความชัดเจน)
  if (!is.na(cat_col)) {
    cc <- dt %>% count(!!sym(cat_col), name = "n") %>% arrange(desc(n))
    plot_list[[1]] <- ggplot(cc, aes(x = "", y = n, fill = !!sym(cat_col))) +
      geom_col(width = 1, colour = "white") +
      coord_polar("y") +
      labs(title = "Events by Category", fill = NULL) +
      theme_void()
  }

  # Events per Month
  if (!is.na(date_col)) {
    dt$.month_year <- format(as.Date(dt[[date_col]]), "%Y-%m")
    mc <- dt %>% count(.month_year) %>% arrange(.month_year)
    plot_list[[2]] <- ggplot(mc, aes(x = .month_year, y = n)) +
      geom_col(fill = "steelblue", colour = "black", size = 0.3) +
      labs(title = "Events per Month", x = "Month", y = "Events Count") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
  }

  if (length(plot_list) > 0) {
    combined <- do.call(grid.arrange, c(plot_list, ncol = 2,
                        list(top = "⏱️ US-Iran War — Event Timeline Analysis")))
    ggsave("figures/04_timeline.png", combined, width = 14, height = 5, dpi = 120)
    message("✅ Saved: figures/04_timeline.png")
  }

  message("\n📋 Full Timeline:")
  print(dt %>% select(-any_of(".month_year")))
}

# --- 11e. Pros & Cons Charts (Cell 9) ---
plot_pros_cons <- function(df_proscons_c) {

  if (is.null(df_proscons_c)) {
    message("⚠️  Pros/Cons DataFrame not available"); return(invisible(NULL))
  }

  dpc        <- df_proscons_c
  type_col   <- names(dpc)[grep("type",   names(dpc), ignore.case = TRUE)][1]
  cat_col    <- names(dpc)[grep("categ",  names(dpc), ignore.case = TRUE)][1]
  impact_col <- names(dpc)[grep("impact", names(dpc), ignore.case = TRUE)][1]
  title_col  <- names(dpc)[grep("title",  names(dpc), ignore.case = TRUE)][1]

  plot_list <- list()

  # Pro vs Con Count
  if (!is.na(type_col)) {
    tc <- dpc %>% count(!!sym(type_col))
    plot_list[[1]] <- ggplot(tc, aes(x = !!sym(type_col), y = n,
                                     fill = !!sym(type_col))) +
      geom_col(colour = "black", size = 0.3) +
      scale_fill_manual(values = c(Pro = "#2ca02c", Con = "#d62728"),
                        na.value = "steelblue") +
      geom_text(aes(label = n), vjust = -0.3, fontface = "bold") +
      labs(title = "Pro vs Con Count", x = NULL, y = "Count") +
      theme_minimal() + theme(legend.position = "none")
  }

  # Category × Type Breakdown
  if (!is.na(cat_col) && !is.na(type_col)) {
    ct <- dpc %>% count(!!sym(cat_col), !!sym(type_col))
    plot_list[[2]] <- ggplot(ct, aes(x = n, y = !!sym(cat_col),
                                     fill = !!sym(type_col))) +
      geom_col(position = "stack", colour = "black", size = 0.3) +
      scale_fill_manual(values = c(Pro = "#2ca02c", Con = "#d62728"),
                        na.value = "steelblue") +
      labs(title = "Category × Type Breakdown",
           x = "Count", y = NULL, fill = "Type") +
      theme_minimal()
  }

  # Impact Level Distribution
  if (!is.na(impact_col)) {
    impact_order  <- c("Critical", "High", "Moderate", "Low")
    impact_colours <- c(Critical = "#d62728", High = "#ff7f0e",
                        Moderate = "#ffbb78", Low = "#2ca02c")
    il <- dpc %>% count(!!sym(impact_col)) %>%
      mutate(!!sym(impact_col) := factor(!!sym(impact_col), levels = impact_order)) %>%
      arrange(!!sym(impact_col))

    plot_list[[3]] <- ggplot(il, aes(x = !!sym(impact_col), y = n,
                                     fill = !!sym(impact_col))) +
      geom_col(colour = "black", size = 0.3) +
      scale_fill_manual(values = impact_colours, na.value = "steelblue") +
      geom_text(aes(label = n), vjust = -0.3, fontface = "bold") +
      labs(title = "Impact Level Distribution", x = NULL, y = "Count") +
      theme_minimal() + theme(legend.position = "none")
  }

  if (length(plot_list) > 0) {
    combined <- do.call(grid.arrange, c(plot_list, ncol = 3,
                        list(top = "✅❌ Pros & Cons of the 2026 Oil Price Surge")))
    ggsave("figures/05_pros_cons.png", combined, width = 18, height = 5, dpi = 120)
    message("✅ Saved: figures/05_pros_cons.png")
  }

  # Summary table
  if (!is.na(type_col) && !is.na(title_col) && !is.na(impact_col)) {
    crit_cons <- dpc %>%
      filter(!!sym(type_col) == "Con", !!sym(impact_col) == "Critical")
    if (nrow(crit_cons) > 0) {
      cat("\n🔴 CRITICAL CONS:\n")
      for (t in crit_cons[[title_col]]) cat(sprintf("  ❌ %s\n", t))
    }
    pros <- dpc %>% filter(!!sym(type_col) == "Pro")
    if (nrow(pros) > 0) {
      cat("\n🟢 NOTABLE PROS:\n")
      for (t in pros[[title_col]]) cat(sprintf("  ✅ %s\n", t))
    }
  }
}

# --- 11f. Model Performance + Residual Charts (Cell 15) ---
plot_model_performance <- function(all_res, y_test) {

  if (length(all_res) == 0) {
    message("❌ No models completed!"); return(invisible(NULL))
  }

  rows <- lapply(all_res, function(r) {
    data.frame(
      Model      = r$model_name,
      Train_R2   = round(r$train_r2,  4),
      Test_R2    = round(r$test_r2,   4),
      Test_RMSE  = round(r$test_rmse, 4),
      Test_MAE   = round(r$test_mae,  4),
      Test_MAPE  = round(r$test_mape, 2),
      CV_R2      = if (!is.null(r$cv_mean)) round(r$cv_mean, 4) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  cdf      <- do.call(rbind, rows) %>% arrange(desc(Test_R2))
  best_res <- all_res[[which.max(cdf$Test_R2)]]

  cat("\n📊 Model Performance Summary:\n")
  print(cdf)
  cat(sprintf("\n🏆 BEST MODEL: %s\n",   best_res$model_name))
  cat(sprintf("   Test R²:   %.4f\n",  best_res$test_r2))
  cat(sprintf("   Test RMSE: %.4f\n",  best_res$test_rmse))
  cat(sprintf("   Test MAPE: %.2f%%\n", best_res$test_mape))

  # Bar charts: R² and RMSE
  p1 <- ggplot(cdf, aes(x = reorder(Model, Test_R2), y = Test_R2)) +
    geom_col(fill = "steelblue", colour = "black", size = 0.3) +
    geom_text(aes(label = sprintf("%.4f", Test_R2)), hjust = -0.1, size = 3) +
    coord_flip() +
    labs(title = "Test R² (higher = better)", x = NULL, y = "Test R²") +
    theme_minimal()

  p2 <- ggplot(cdf, aes(x = reorder(Model, -Test_RMSE), y = Test_RMSE)) +
    geom_col(fill = "steelblue", colour = "black", size = 0.3) +
    geom_text(aes(label = sprintf("%.3f", Test_RMSE)), hjust = -0.1, size = 3) +
    coord_flip() +
    labs(title = "Test RMSE (lower = better)", x = NULL, y = "Test RMSE") +
    theme_minimal()

  combined1 <- grid.arrange(p1, p2, ncol = 2,
                             top = "Model Performance Comparison")
  ggsave("figures/06_model_performance.png", combined1, width = 14, height = 5, dpi = 120)

  # Actual vs Predicted + Residuals
  y_pred_best <- best_res$predictions
  resid_df    <- data.frame(actual = y_test, predicted = y_pred_best)

  p3 <- ggplot(resid_df, aes(x = actual, y = predicted)) +
    geom_point(colour = "steelblue", size = 3, alpha = 0.8) +
    geom_abline(slope = 1, intercept = 0, colour = "red", linetype = "dashed", size = 1) +
    labs(title = "Actual vs Predicted",
         x = "Actual % Increase", y = "Predicted % Increase") +
    theme_minimal()

  p4 <- ggplot(resid_df, aes(x = actual - predicted)) +
    geom_histogram(fill = "coral", colour = "white", alpha = 0.85, bins = 10) +
    geom_vline(xintercept = 0, colour = "black", linetype = "dashed", size = 1) +
    labs(title = "Residual Distribution", x = "Residuals", y = "Count") +
    theme_minimal()

  combined2 <- grid.arrange(p3, p4, ncol = 2,
                             top = sprintf("Residual Analysis — %s", best_res$model_name))
  ggsave("figures/07_residuals.png", combined2, width = 12, height = 5, dpi = 120)

  message("✅ Saved: figures/06_model_performance.png, figures/07_residuals.png")
  return(invisible(cdf))
}

# --- 11g. Feature Importance (Cell 16) ---
plot_feature_importance <- function(best_res, feature_cols) {

  if (is.null(best_res)) return(invisible(NULL))

  model_obj <- best_res$model

  # Tree-based: feature_importances_
  if (inherits(model_obj, "randomForest")) {
    imp <- importance(model_obj)
    imp_df <- data.frame(
      Feature    = rownames(imp),
      Importance = imp[, 1],
      stringsAsFactors = FALSE
    ) %>% arrange(desc(Importance))

    top_n <- min(15, nrow(imp_df))
    p <- ggplot(imp_df[1:top_n, ],
                aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_col(fill = "steelblue", colour = "black", size = 0.3) +
      coord_flip() +
      labs(title = sprintf("Top %d Feature Importances — %s", top_n, best_res$model_name),
           x = NULL, y = "Importance Score") +
      theme_minimal()

    ggsave("figures/08_feature_importance.png", p, width = 10, height = 6, dpi = 120)
    message("✅ Saved: figures/08_feature_importance.png")
    cat("\nTop features:\n"); print(imp_df)

  } else {
    message("⚠️  Feature importance not available for this model type")
  }
}

# --- 11h. Global Dashboard (Cell 18) ---
plot_dashboard <- function(df_crude_c, df_petrol_c, df_country_c, best_res, y_test) {

  plots <- list()

  # Panel 1: Brent Price Trajectory
  if (!is.null(df_crude_c)) {
    dc       <- df_crude_c
    date_col <- names(dc)[grep("date",  names(dc), ignore.case = TRUE)][1]
    num_cols_d <- names(dc)[sapply(dc, is.numeric)]
    bc_col   <- intersect(
      grep("brent", num_cols_d, ignore.case = TRUE, value = TRUE),
      grep("usd",   num_cols_d, ignore.case = TRUE, value = TRUE)
    )[1]
    if (!is.na(bc_col)) {
      if (!is.na(date_col)) dc <- dc %>% arrange(!!sym(date_col))
      dc$.idx <- seq_len(nrow(dc))
      plots[[1]] <- ggplot(dc, aes(x = .idx, y = !!sym(bc_col))) +
        geom_line(colour = "#d62728", size = 1.2) +
        geom_point(colour = "#d62728", size = 2) +
        labs(title = "Brent Price Trajectory", x = NULL, y = "USD/barrel") +
        theme_minimal()
    }
  }

  # Panel 2: Brent % Change by Phase
  if (!is.null(df_crude_c)) {
    phase_c  <- names(df_crude_c)[grep("phase", names(df_crude_c), ignore.case = TRUE)][1]
    num_cols_cr <- names(df_crude_c)[sapply(df_crude_c, is.numeric)]
    bc_chg   <- intersect(
      grep("brent", num_cols_cr, ignore.case = TRUE, value = TRUE),
      grep("pct",   num_cols_cr, ignore.case = TRUE, value = TRUE)
    )[1]
    if (!is.na(phase_c) && !is.na(bc_chg)) {
      grp <- df_crude_c %>%
        group_by(!!sym(phase_c)) %>%
        summarise(mean_chg = mean(!!sym(bc_chg), na.rm = TRUE), .groups = "drop") %>%
        arrange(desc(mean_chg)) %>%
        mutate(.colour = ifelse(grepl("Active", !!sym(phase_c)), "#d62728", "steelblue"))
      plots[[2]] <- ggplot(grp, aes(x = reorder(!!sym(phase_c), mean_chg), y = mean_chg,
                                    fill = .colour)) +
        geom_col(colour = "black", size = 0.3) +
        scale_fill_identity() +
        coord_flip() +
        labs(title = "Mean Daily % Change by Phase", x = NULL, y = "Mean % Change") +
        theme_minimal()
    }
  }

  # Panel 3: Petrol % Increase by Country
  if (!is.null(df_petrol_c)) {
    dp_   <- df_petrol_c
    c_col <- names(dp_)[grep("country", names(dp_), ignore.case = TRUE)][1]
    p_col <- names(dp_)[grep("pct",     names(dp_), ignore.case = TRUE)][1]
    if (!is.na(c_col) && !is.na(p_col)) {
      ds_ <- dp_ %>% arrange(!!sym(p_col)) %>%
        mutate(.col = case_when(
          !!sym(p_col) > 5 ~ "#d62728",
          !!sym(p_col) > 0 ~ "#ff7f0e",
          TRUE              ~ "#2ca02c"
        ))
      plots[[3]] <- ggplot(ds_, aes(x = reorder(!!sym(c_col), !!sym(p_col)),
                                    y = !!sym(p_col), fill = .col)) +
        geom_col(colour = "black", size = 0.3) +
        scale_fill_identity() +
        coord_flip() +
        labs(title = "Petrol % Increase by Country", x = NULL, y = "% Increase") +
        theme_minimal()
    }
  }

  # Panel 4: GDP Impact by Country
  if (!is.null(df_country_c)) {
    dc2      <- df_country_c
    gdp_col2 <- names(dc2)[grep("gdp",     names(dc2), ignore.case = TRUE)][1]
    cty_col2 <- names(dc2)[grep("country", names(dc2), ignore.case = TRUE)][1]
    if (!is.na(gdp_col2) && !is.na(cty_col2)) {
      ds2 <- dc2 %>% arrange(!!sym(gdp_col2)) %>%
        mutate(.col2 = ifelse(!!sym(gdp_col2) < 0, "#d62728", "#2ca02c"))
      plots[[4]] <- ggplot(ds2, aes(x = reorder(!!sym(cty_col2), !!sym(gdp_col2)),
                                    y = !!sym(gdp_col2), fill = .col2)) +
        geom_col(colour = "black", size = 0.3) +
        scale_fill_identity() +
        coord_flip() +
        labs(title = "GDP Impact by Country", x = NULL, y = "GDP Impact (%)") +
        theme_minimal()
    }
  }

  # Panel 5: Actual vs Predicted
  if (!is.null(best_res)) {
    y_pred  <- best_res$predictions
    res_df  <- data.frame(actual = y_test, predicted = y_pred)
    plots[[5]] <- ggplot(res_df, aes(x = actual, y = predicted)) +
      geom_point(colour = "steelblue", size = 3, alpha = 0.8) +
      geom_abline(slope = 1, intercept = 0, colour = "red",
                  linetype = "dashed", size = 1) +
      labs(title = sprintf("Best Model: %s\nActual vs Predicted", best_res$model_name),
           x = "Actual", y = "Predicted") +
      theme_minimal()
  }

  n <- length(plots)
  if (n > 0) {
    ncols    <- min(3, n)
    combined <- do.call(grid.arrange, c(plots, ncol = ncols,
                        list(top = "🌐 2026 US-Iran War — Global Oil Crisis Dashboard")))
    ggsave("figures/09_dashboard.png", combined, width = 18, height = 12, dpi = 120)
    message("✅ Saved: figures/09_dashboard.png")
  }
}

# ============================================================================
# 12. ฟังก์ชันหลัก (Main)
# ============================================================================

main <- function(data_dir = "data") {

  cat(rep("=", 80), "\n", sep = "")
  cat("🚀 IMPACT ANALYSIS: 2026 US-Iran War Oil Price Impact Analysis\n")
  cat(rep("=", 80), "\n\n", sep = "")

  # สร้างโฟลเดอร์ figures
  if (!dir.exists("figures")) dir.create("figures")

  # ── Step 1: Load ─────────────────────────────────────────────────────────
  cat("📥 Step 1: Loading data...\n")
  dfs <- load_data_smart(data_dir)
  df_crude    <- dfs$crude
  df_petrol   <- dfs$petrol
  df_country  <- dfs$country
  df_proscons <- dfs$pros_cons
  df_timeline <- dfs$timeline

  # ── Step 2: Audit ─────────────────────────────────────────────────────────
  cat("\n📊 Step 2: Auditing data...\n")
  audit_data(df_crude,    "crude_oil_daily")
  audit_data(df_petrol,   "petrol_prices")
  audit_data(df_country,  "country_impact")
  audit_data(df_proscons, "pros_cons")
  audit_data(df_timeline, "war_timeline")
  cat("✅ Audit complete\n")

  # ── Step 3: Clean ─────────────────────────────────────────────────────────
  cat("\n🧹 Step 3: Cleaning data...\n")
  df_crude_c    <- clean_df(df_crude,    "crude_oil_daily")
  df_petrol_c   <- clean_df(df_petrol,   "petrol_prices")
  df_country_c  <- clean_df(df_country,  "country_impact")
  df_proscons_c <- clean_df(df_proscons, "pros_cons")
  df_timeline_c <- clean_df(df_timeline, "war_timeline")
  cat("\n✅ ALL DATA CLEANING COMPLETE — ZERO NaN GUARANTEED\n")

  # ── Step 4: EDA Visualizations ────────────────────────────────────────────
  cat("\n📊 Step 4: EDA Visualizations...\n")
  plot_crude_oil(df_crude_c)
  plot_petrol_prices(df_petrol_c)
  plot_country_impact(df_country_c)
  plot_timeline(df_timeline_c)
  plot_pros_cons(df_proscons_c)

  # ── Step 5: Feature Engineering ───────────────────────────────────────────
  cat("\n⚙️ Step 5: Building features...\n")
  if (is.null(df_petrol_c)) stop("❌ petrol data required for modeling")

  df_country_for_merge <- if (!is.null(df_country_c)) df_country_c else
    data.frame(Country = character(0))

  feat_result  <- build_features(df_petrol_c, df_country_for_merge)
  df_merged    <- feat_result$df
  TARGET       <- feat_result$target
  feature_cols <- select_features(df_merged, TARGET)

  X <- df_merged[, feature_cols, drop = FALSE]
  y <- df_merged[[TARGET]]

  # ── Step 6: Quality Check ─────────────────────────────────────────────────
  cat("\n✅ Step 6: Quality check...\n")
  qc_result    <- quality_check(X, y)
  X <- qc_result$X
  y <- qc_result$y

  # ── Step 7: Train-Test Split ──────────────────────────────────────────────
  cat("\n📊 Step 7: Train-Test Split...\n")
  TEST_SIZE <- if (nrow(X) > 20) 0.2 else 0.15
  cat(sprintf("📊 Train-Test Split  (test_size=%.2f)\n%s\n",
              TEST_SIZE, paste(rep("-", 50), collapse = "")))

  set.seed(RANDOM_STATE)
  n_test   <- max(1, round(nrow(X) * TEST_SIZE))
  test_idx <- sample(seq_len(nrow(X)), n_test)
  train_idx <- setdiff(seq_len(nrow(X)), test_idx)

  X_train <- X[train_idx, , drop = FALSE]
  X_test  <- X[test_idx,  , drop = FALSE]
  y_train <- y[train_idx]
  y_test  <- y[test_idx]

  stopifnot(nrow(X_train) == length(y_train), nrow(X_test) == length(y_test))
  cat(sprintf("✅ Train: %d samples  |  Test: %d samples\n",
              nrow(X_train), nrow(X_test)))

  # Scaling สำหรับ Ridge/Lasso
  sc       <- scale_features(X, X_train, X_test)
  X_train_sc <- sc$X_train_sc
  X_test_sc  <- sc$X_test_sc

  # ── Step 8: Train Models ──────────────────────────────────────────────────
  cat("\n🤖 Step 8: Training models...\n")

  # Model 1: Ridge
  ridge_res <- tryCatch({
    m <- glmnet::glmnet(as.matrix(X_train_sc), y_train, alpha = 0, lambda = 1.0)
    evaluate_model_glmnet(m, X_train_sc, X_test_sc, y_train, y_test,
                          "Ridge Regression", lambda = 1.0)
  }, error = function(e) { message("❌ Ridge failed: ", e$message); NULL })

  # Model 2: Lasso
  lasso_res <- tryCatch({
    m <- glmnet::glmnet(as.matrix(X_train_sc), y_train, alpha = 1, lambda = 0.5)
    evaluate_model_glmnet(m, X_train_sc, X_test_sc, y_train, y_test,
                          "Lasso Regression", lambda = 0.5)
  }, error = function(e) { message("❌ Lasso failed: ", e$message); NULL })

  # Model 3: Random Forest
  rf_res <- tryCatch({
    m <- randomForest(x = X_train, y = y_train,
                      ntree = 200, max.depth = 5, importance = TRUE)
    evaluate_model_rf(m, X_train, X_test, y_train, y_test, "Random Forest")
  }, error = function(e) { message("❌ RF failed: ", e$message); NULL })

  # Model 4: Gradient Boosting
  gb_res <- tryCatch({
    df_tr <- cbind(X_train, .y_target = y_train)
    m <- gbm::gbm(.y_target ~ ., data = df_tr, distribution = "gaussian",
                  n.trees = 100, interaction.depth = 3,
                  shrinkage = 0.1, verbose = FALSE)
    evaluate_model_gbm(m, X_train, X_test, y_train, y_test,
                       "Gradient Boosting", n.trees = 100)
  }, error = function(e) { message("❌ GB failed: ", e$message); NULL })

  # Model 5: XGBoost (optional)
  xgb_res <- NULL
  if (XGB_OK) {
    xgb_res <- tryCatch({
      dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
      dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test)
      params <- list(max_depth = 3, learning_rate = 0.1, verbosity = 0,
                     objective = "reg:squarederror")
      m <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
      evaluate_model_xgb(m, dtrain, dtest, y_train, y_test, "XGBoost")
    }, error = function(e) { message("❌ XGBoost failed: ", e$message); NULL })
  }

  # Model 6: LightGBM (optional)
  lgb_res <- NULL
  if (LGB_OK) {
    lgb_res <- tryCatch({
      dtrain <- lgb.Dataset(as.matrix(X_train), label = y_train)
      params <- list(num_leaves = 15, learning_rate = 0.1, verbosity = -1,
                     objective = "regression", n_jobs = -1)
      m <- lgb.train(params = params, data = dtrain, nrounds = 100, verbose = -1)
      evaluate_model_lgb(m, X_train, X_test, y_train, y_test, "LightGBM")
    }, error = function(e) { message("❌ LightGBM failed: ", e$message); NULL })
  }

  all_res <- Filter(Negate(is.null),
                    list(ridge_res, lasso_res, rf_res, gb_res, xgb_res, lgb_res))
  message(sprintf("\n✅ Successful models: %d", length(all_res)))

  # ── Step 9: Performance Summary & Visualization ──────────────────────────
  cat("\n📊 Step 9: Performance Summary...\n")
  best_res <- NULL
  if (length(all_res) > 0) {
    cdf      <- plot_model_performance(all_res, y_test)
    best_res <- all_res[[which.max(sapply(all_res, function(r) r$test_r2))]]

    # Feature Importance
    if (!is.null(rf_res)) plot_feature_importance(rf_res, feature_cols)
  }

  # ── Step 10: Global Dashboard ─────────────────────────────────────────────
  cat("\n📊 Step 10: Global Dashboard...\n")
  plot_dashboard(df_crude_c, df_petrol_c, df_country_c, best_res, y_test)

  cat("\n", rep("=", 80), "\n", sep = "")
  cat("✅ Analysis Complete!\n")
  cat(rep("=", 80), "\n\n", sep = "")

  return(list(
    dfs          = list(crude = df_crude_c, petrol = df_petrol_c,
                        country = df_country_c, pros_cons = df_proscons_c,
                        timeline = df_timeline_c),
    df_merged    = df_merged,
    feature_cols = feature_cols,
    TARGET       = TARGET,
    all_res      = all_res,
    best_res     = best_res
  ))
}

# ── Thin wrappers สำหรับ evaluate_model ────────────────────────────────────

evaluate_model_glmnet <- function(m, X_tr, X_te, y_tr, y_te, name, lambda) {
  cat(sprintf("\n%s\n🔍 %s\n%s\n",
              paste(rep("=", 65), collapse = ""), name,
              paste(rep("=", 65), collapse = "")))
  tryCatch({
    y_tr_p <- as.numeric(predict(m, as.matrix(X_tr), s = lambda))
    y_te_p <- as.numeric(predict(m, as.matrix(X_te), s = lambda))
    .make_result(m, name, y_tr, y_te, y_tr_p, y_te_p, X_tr)
  }, error = function(e) { cat(sprintf("❌ %s failed: %s\n", name, e$message)); NULL })
}

evaluate_model_rf <- function(m, X_tr, X_te, y_tr, y_te, name) {
  cat(sprintf("\n%s\n🔍 %s\n%s\n",
              paste(rep("=", 65), collapse = ""), name,
              paste(rep("=", 65), collapse = "")))
  tryCatch({
    y_tr_p <- predict(m, X_tr)
    y_te_p <- predict(m, X_te)
    .make_result(m, name, y_tr, y_te, y_tr_p, y_te_p, X_tr)
  }, error = function(e) { cat(sprintf("❌ %s failed: %s\n", name, e$message)); NULL })
}

evaluate_model_gbm <- function(m, X_tr, X_te, y_tr, y_te, name, n.trees) {
  cat(sprintf("\n%s\n🔍 %s\n%s\n",
              paste(rep("=", 65), collapse = ""), name,
              paste(rep("=", 65), collapse = "")))
  tryCatch({
    y_tr_p <- predict(m, cbind(X_tr, .y_target = y_tr), n.trees = n.trees)
    y_te_p <- predict(m, cbind(X_te, .y_target = y_te), n.trees = n.trees)
    .make_result(m, name, y_tr, y_te, y_tr_p, y_te_p, X_tr)
  }, error = function(e) { cat(sprintf("❌ %s failed: %s\n", name, e$message)); NULL })
}

evaluate_model_xgb <- function(m, dtrain, dtest, y_tr, y_te, name) {
  cat(sprintf("\n%s\n🔍 %s\n%s\n",
              paste(rep("=", 65), collapse = ""), name,
              paste(rep("=", 65), collapse = "")))
  tryCatch({
    y_tr_p <- predict(m, dtrain)
    y_te_p <- predict(m, dtest)
    .make_result(m, name, y_tr, y_te, y_tr_p, y_te_p, NULL)
  }, error = function(e) { cat(sprintf("❌ %s failed: %s\n", name, e$message)); NULL })
}

evaluate_model_lgb <- function(m, X_tr, X_te, y_tr, y_te, name) {
  cat(sprintf("\n%s\n🔍 %s\n%s\n",
              paste(rep("=", 65), collapse = ""), name,
              paste(rep("=", 65), collapse = "")))
  tryCatch({
    y_tr_p <- predict(m, as.matrix(X_tr))
    y_te_p <- predict(m, as.matrix(X_te))
    .make_result(m, name, y_tr, y_te, y_tr_p, y_te_p, NULL)
  }, error = function(e) { cat(sprintf("❌ %s failed: %s\n", name, e$message)); NULL })
}

.make_result <- function(model, name, y_tr, y_te, y_tr_p, y_te_p, X_tr) {
  tr_r2   <- if (sd(y_tr_p) > 0) cor(y_tr, y_tr_p)^2 else 0
  te_r2   <- if (sd(y_te_p) > 0) cor(y_te, y_te_p)^2 else 0
  tr_rmse <- sqrt(mean((y_tr - y_tr_p)^2))
  te_rmse <- sqrt(mean((y_te - y_te_p)^2))
  tr_mae  <- mean(abs(y_tr - y_tr_p))
  te_mae  <- mean(abs(y_te - y_te_p))
  te_mape <- mean(abs((y_te - y_te_p) / (abs(y_te) + 1e-10))) * 100

  cat(sprintf("%-22s %-14s %-14s\n", "Metric", "Train", "Test"))
  cat(paste(rep("-", 50), collapse = ""), "\n")
  cat(sprintf("%-22s %-14.4f %-14.4f\n", "R²",        tr_r2,   te_r2))
  cat(sprintf("%-22s %-14.4f %-14.4f\n", "RMSE",      tr_rmse, te_rmse))
  cat(sprintf("%-22s %-14.4f %-14.4f\n", "MAE",       tr_mae,  te_mae))
  cat(sprintf("%-22s %-14s %-14.2f\n",   "MAPE (%)",  "-",     te_mape))
  gap <- abs(tr_r2 - te_r2)
  cat(sprintf("\nOverfitting gap: %.4f  %s\n",
              gap, if (gap < 0.15) "✅ OK" else "⚠️ Review"))

  list(
    model_name = name, model = model, predictions = y_te_p,
    problem_type = "Regression",
    train_r2 = tr_r2,   test_r2 = te_r2,
    train_rmse = tr_rmse, test_rmse = te_rmse,
    train_mae = tr_mae,   test_mae = te_mae,
    test_mape = te_mape,  cv_mean = NULL, cv_std = NULL
  )
}

# ============================================================================
# 13. รันโปรแกรมหลัก
# ============================================================================

# เรียกใช้งาน:
# results <- main(data_dir = "data")

message("✅ Script พร้อมใช้งาน!")
```

```{r run_analysis}
# แก้ path ให้ตรงกับโฟลเดอร์ข้อมูลของคุณ
results <- main(data_dir = "d:/Visul Studio/Global Petrol Prices — Impact of 2026 US-Iran War")
```
