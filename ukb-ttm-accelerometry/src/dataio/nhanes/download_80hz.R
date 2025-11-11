#!/usr/bin/env Rscript
# Download NHANES raw 80 Hz accelerometry data (2011-2014)
# Uses NHANES.RAW80Hz package to programmatically fetch data

suppressPackageStartupMessages({
  library(optparse)
})

# Parse command line arguments
option_list <- list(
  make_option(c("--out"), type="character", default="data/nhanes/raw80",
              help="Output directory for raw 80Hz data [default: %default]"),
  make_option(c("--cycles"), type="character", default="2011-2012,2013-2014",
              help="NHANES cycles to download (comma-separated) [default: %default]"),
  make_option(c("--participants"), type="character", default=NULL,
              help="Specific participant IDs (comma-separated, optional)"),
  make_option(c("--mirror"), type="character", default=NULL,
              help="Local mirror directory (optional)"),
  make_option(c("--verbose"), action="store_true", default=FALSE,
              help="Verbose output"),
  make_option(c("--install-pkg"), action="store_true", default=FALSE,
              help="Install RNHANES package if missing")
)

opt_parser <- OptionParser(option_list=option_list,
                          description="Download NHANES raw 80Hz accelerometry data")
opt <- parse_args(opt_parser)

# Function to install and load required packages
install_and_load <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing package: %s\n", pkg))
    install.packages(pkg, repos="https://cloud.r-project.org", quiet = !opt$verbose)
    library(pkg, character.only = TRUE)
  }
}

# Check/install required packages
if (opt$install_pkg || !require("RNHANES", quietly = TRUE)) {
  cat("Installing required packages...\n")
  install_and_load("RNHANES")
}

# Load required packages
suppressPackageStartupMessages({
  library(RNHANES)
})

cat("========================================\n")
cat("NHANES 80Hz Data Downloader\n")
cat("========================================\n\n")

# Parse cycles
cycles <- strsplit(opt$cycles, ",")[[1]]
cycles <- trimws(cycles)

cat(sprintf("Cycles to download: %s\n", paste(cycles, collapse=", ")))
cat(sprintf("Output directory: %s\n", opt$out))

# Create output directory
dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)

# Parse participant IDs if specified
participant_ids <- NULL
if (!is.null(opt$participants)) {
  participant_ids <- strsplit(opt$participants, ",")[[1]]
  participant_ids <- trimws(participant_ids)
  cat(sprintf("Downloading %d specific participants\n", length(participant_ids)))
}

# Initialize manifest
manifest <- data.frame(
  participant_id = character(),
  cycle = character(),
  device_location = character(),
  start_time_utc = character(),
  end_time_utc = character(),
  path = character(),
  stringsAsFactors = FALSE
)

# Download data for each cycle
for (cycle in cycles) {
  cat(sprintf("\n--- Processing cycle: %s ---\n", cycle))

  # Create cycle directory
  cycle_dir <- file.path(opt$out, cycle)
  dir.create(cycle_dir, recursive = TRUE, showWarnings = FALSE)

  # Get list of accelerometry files for this cycle
  # Note: The actual NHANES.RAW80Hz package may have different function names
  # This is a template - adjust based on actual package API

  tryCatch({
    # Download demographic data to get participant list
    demo_file <- sprintf("DEMO_%s", gsub("-", "_", cycle))

    if (opt$verbose) {
      cat(sprintf("Fetching demographic file: %s\n", demo_file))
    }

    demo_data <- nhanes(demo_file)

    if (is.null(demo_data)) {
      cat(sprintf("Warning: Could not fetch demographics for %s\n", cycle))
      next
    }

    # Get list of participants with accelerometry data
    # NHANES accelerometry file prefix depends on cycle
    if (cycle == "2011-2012") {
      pax_file <- "PAXRAW_G"
    } else if (cycle == "2013-2014") {
      pax_file <- "PAXRAW_H"
    } else {
      cat(sprintf("Warning: Unknown cycle format: %s\n", cycle))
      next
    }

    # Attempt to get accelerometry data
    # Note: Actual implementation depends on NHANES.RAW80Hz package
    # This is a simplified version

    if (opt$verbose) {
      cat(sprintf("Fetching accelerometry file: %s\n", pax_file))
    }

    pax_data <- nhanes(pax_file)

    if (is.null(pax_data)) {
      cat(sprintf("Warning: No accelerometry data for %s\n", cycle))
      next
    }

    # Get unique participant IDs
    if ("SEQN" %in% names(pax_data)) {
      available_ids <- unique(pax_data$SEQN)
    } else {
      cat("Warning: SEQN column not found\n")
      next
    }

    # Filter to requested participants if specified
    if (!is.null(participant_ids)) {
      available_ids <- intersect(available_ids, as.numeric(participant_ids))
    }

    cat(sprintf("Found %d participants with accelerometry data\n", length(available_ids)))

    # Download each participant's data
    for (pid in available_ids) {
      if (opt$verbose) {
        cat(sprintf("Downloading participant %s...\n", pid))
      }

      # Create participant directory
      pid_dir <- file.path(cycle_dir, as.character(pid))
      dir.create(pid_dir, recursive = TRUE, showWarnings = FALSE)

      # Save raw data (actual format depends on NHANES structure)
      output_file <- file.path(pid_dir, sprintf("%s_raw80.RData", pid))

      # Extract participant's accelerometry data
      participant_data <- pax_data[pax_data$SEQN == pid, ]

      # Save to file
      save(participant_data, file = output_file)

      # Add to manifest
      # Note: Actual start/end times need to be extracted from data
      manifest <- rbind(manifest, data.frame(
        participant_id = as.character(pid),
        cycle = cycle,
        device_location = "hip",  # NHANES 2011-2014 typically hip-worn
        start_time_utc = NA,  # Extract from data
        end_time_utc = NA,    # Extract from data
        path = output_file,
        stringsAsFactors = FALSE
      ))
    }

  }, error = function(e) {
    cat(sprintf("Error processing cycle %s: %s\n", cycle, e$message))
  })
}

# Save manifest
manifest_path <- file.path(opt$out, "manifest.csv")
write.csv(manifest, manifest_path, row.names = FALSE)

cat(sprintf("\n========================================\n"))
cat(sprintf("Download complete!\n"))
cat(sprintf("Total participants: %d\n", nrow(manifest)))
cat(sprintf("Manifest saved to: %s\n", manifest_path))
cat(sprintf("========================================\n"))

cat("\nNEXT STEPS:\n")
cat("1. Convert to parquet:\n")
cat(sprintf("   python -m src.dataio.nhanes.convert_80hz_to_parquet --input %s\n", opt$out))
cat("\n2. Parse to windows:\n")
cat("   python -m src.dataio.nhanes.parse_80hz --input data/nhanes/80hz_parquet\n")
