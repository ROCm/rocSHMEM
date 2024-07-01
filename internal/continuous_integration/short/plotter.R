#!/usr/bin/env Rscript

# load the required libraries:
library(tidyverse)
library(RColorBrewer)
library(optparse)

# declare some helper functions
ggpreview <- function (..., device = "png") {
  fname <- tempfile(fileext = paste0(".", device))
  ggplot2::ggsave(filename = fname, device = device, ...)
  system2("open", fname)
  invisible(NULL)
}

set_right_order <- function(df) {
  # reverse the order of the rows so that oldest commit is first
  df <- df %>% map_df(rev)
  # ensure that ggplot plots the x-axis in the right order
  df$Commit <- factor(df$Commit, levels = unique(df$Commit))
  return(df)
}

plot_and_save <- function(df, xval, yval, title, subtitle, xlabel, filename) {
  p <- ggplot(df, aes_string(x=xval, y=yval, group=1)) +
    geom_line(size = 0.5, color=mycolors[1]) +
    geom_point(size = 1.5, alpha = 1, color=mycolors[2]) +
    theme_minimal() +
    expand_limits(y=0) +
    xlab(xlabel) +
    ggtitle(title, subtitle = subtitle) +
    theme(
      axis.text.x = element_text(angle=90,hjust=1),
      axis.title.y = element_blank()
    ) +
  scale_fill_manual(values = mycolors)
  #ggpreview(width=7.5, height=5, units="in", dpi=500)
  ggsave(filename, p, device=pdf, dpi=500)
}

## Set up options ##

option_list = list(
    make_option(c("-o", "--output"), type="character", default=NULL, action="store",
                   help="path (without trailing /) to a folder that will
                   contain the plots", metavar="folder-path"),
    make_option(c("-a", "--changeset_a"), type="character", default=NULL, action="store",
                   help="beginning (inclusive) changeset of slice", metavar="changeset"),
    make_option(c("-b", "--changeset_b"), type="character", default=NULL, action="store",
                   help="ending (inclusive) changeset of slice", metavar="changeset"),
    make_option(c("-c", "--one_changeset"), type="character", default=NULL, action="store",
                help="if set, will prepare plots for one changeset; if not, plots for a changeset slice")

)

## SCRIPT START ##

# parse the options
opt_parser <- OptionParser(option_list=option_list)
opts <- parse_args(opt_parser)
if (is.null(opts$output)) {
    print_help(opt_parser)
    stop("Please set the --output flag.", call.=FALSE)
}
slice_opt = 0
single_opt = 0
if (!is.null(opts$changeset_a) && !is.null(opts$changeset_b)) {
    slice_opt = 1
}
if (!is.null(opts$one_changeset)) {
    single_opt = 1
}

if ( (slice_opt && single_opt) || (!slice_opt && !single_opt) ) {
    stop("Please supply a slice or a single changeset, not both.", call.= FALSE)
}

# choose color palette
mycolors <- brewer.pal(5, "Set2")

if (length(opts$one_changeset) > 0) {
  ## Plotting data for a single changeset ##

  # read the files
  non_amo   <- read.csv("non_amo_one_changeset.csv", header=TRUE)
  amo       <- read.csv("amo_one_changeset.csv", header=TRUE)
  ping_pong <- read.csv("ping_pong_one_changeset.csv", header=TRUE)

  # ensure that ggplot plots the x-axis in the right order
  non_amo$size <- factor(non_amo$size, levels = unique(non_amo$size))
  amo$op <- factor(amo$op, levels = unique(amo$op))

  # plot
  non_amo_ops <- list("put","put_nbi","get","get_nbi")
  for (op in non_amo_ops) {
    plot_and_save(df=non_amo,
                  xval="size",
                  yval=op,
                  title=op,
                  subtitle="Latency (us)",
                  xlabel="Message size (bytes)",
                  filename=paste(opts$output,"/",op,"_changeset_",opts$one_changeset,".pdf", sep="")
                  )
  }

  # prepare data for plots with fixed message size and ops as x axis
  non_amo$bsize <- paste("b",non_amo$size,sep="") # (so that the columns in non_amo_t start with a character)
  non_amo_t <- setNames(data.frame(t(non_amo[,2:5])), non_amo[,6]) # transpose + set column names
  non_amo_t$op <- colnames(non_amo[,2:5]) # make a column with operation names

  sizes <- colnames(non_amo_t[,-(length(colnames(non_amo_t)))])
  for (size in sizes) {
    plot_and_save(df=non_amo_t,
                  xval="op",
                  yval=size,
                  title=paste(sub('.', '', size),"byte"),
                  subtitle="Latency (us)",
                  xlabel="Operation",
                  filename=paste(opts$output,"/",size,"_changeset_",opts$one_changeset,".pdf", sep="")
    )
  }

  plot_and_save(df=amo,
                xval="op",
                yval="latency",
                title="Atomics",
                subtitle="Latency (us)",
                xlabel="Operation",
                filename=paste(opts$output,"/atomic_changeset_",opts$one_changeset,".pdf", sep="")
  )

  ping_pong$type <- c("ping_pong")
  p<-ggplot(ping_pong, aes(x=type, y=latency, fill=type)) +
    geom_bar(stat="identity", width=0.5) +
    theme_minimal() +
    ggtitle("Ping pong", subtitle = "Latency (us)") +
    theme(
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.title.x = element_blank(),
      legend.position = "none"
      ) +
    coord_flip() +
    scale_fill_manual(values = mycolors)
  #ggpreview(width=7.5, height=5, units="in", dpi=500)
  ggsave(paste(opts$output,"/ping_pong_changeset_",opts$one_changeset,".pdf", sep=""), p, device=pdf, dpi=500)

} else {
  ## Plotting across a changeset slice ##

  # read the files
  put             <- read.csv("put.csv", header=TRUE)
  put_nbi         <- read.csv("put_nbi.csv", header=TRUE)
  get             <- read.csv("get.csv", header=TRUE)
  get_nbi         <- read.csv("get_nbi.csv", header=TRUE)
  amo             <- read.csv("amo.csv", header=TRUE)
  ping_pong       <- read.csv("ping_pong.csv", header=TRUE)

  # slice out the commits
  start   <- match(c(opts$changeset_a), put$Commit)
  end     <- match(c(opts$changeset_b), put$Commit)
  # (start and end should be the same for all the frames) #
  put             <- put[start:end,]
  put_nbi         <- put_nbi[start:end,]
  get             <- get[start:end,]
  get_nbi         <- get_nbi[start:end,]
  amo             <- amo[start:end,]
  ping_pong       <- ping_pong[start:end,]

  put             <- set_right_order(put)
  put_nbi         <- set_right_order(put_nbi)
  get             <- set_right_order(get)
  get_nbi         <- set_right_order(get_nbi)
  amo             <- set_right_order(amo)
  ping_pong       <- set_right_order(ping_pong)

  # plot
  non_amo_ops <- list("put","put_nbi","get","get_nbi")
  sizes_to_subtitle_map <- list("b1"="1 byte",
                                "b2"="2 bytes",
                                "b4"="4 bytes",
                                "b8"="8 bytes",
                                "b16"="16 bytes",
                                "b32"="32 bytes",
                                "b64"="64 bytes",
                                "b128"="128 bytes",
                                "b256"="256 bytes",
                                "b512"="512 bytes",
                                "b1024"="1024 bytes",
                                "b2048"="2048 bytes",
                                "b4096"="4096 bytes",
                                "b8192"="8192 bytes",
                                "b16384"="16384 bytes",
                                "b32768"="32768 bytes")
  for (op in non_amo_ops) {
    for (size in names(sizes_to_subtitle_map)) {
      plot_and_save(df=eval(parse(text=op)),
                    xval="Commit",
                    yval=size,
                    title=op,
                    subtitle=paste("Latency (us) for ",sizes_to_subtitle_map[[size]],sep=""),
                    xlabel="Commit (older to newer)",
                    filename=paste(opts$output,"/",op,"_",size,".pdf", sep="")
      )
      }
  }

  amo_ops <- list("add","cswap","fadd","fcswap","fetch","finc","inc")
  for (op in amo_ops) {
    plot_and_save(df=amo,
                  xval="Commit",
                  yval=op,
                  title=op,
                  subtitle="Latency (us)",
                  xlabel="Commit (older to newer)",
                  filename=paste(opts$output,"/",op,".pdf", sep="")
    )
  }

  plot_and_save(df=ping_pong,
                xval="Commit",
                yval="latency",
                title="ping_pong",
                subtitle="Latency (us)",
                xlabel="Commit (older to newer)",
                filename=paste(opts$output,"/","ping_pong.pdf", sep="")
  )
}

## SCRIPT END ##
