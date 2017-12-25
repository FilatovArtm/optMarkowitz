library(quantmod)
library(plyr)

start = '2014-03-31'
end = '2015-12-31'

codes = read.csv('stocks.csv', encoding = 'UTF-8', sep=',')[,2]

get_prices <- function(x, start, end, source){
  return = c()
  tryCatch({
  prices = getSymbols(Symbols = as.character(x),from = start, to = end,
                     auto.assign = FALSE, warnings = FALSE, src = source)[,4]
  return = diff(prices)/head(prices,-1)

  }, error=function(e){cat("ERROR :",conditionMessage(e), "\n") })
  return(return)
}

#get yahoo
all_stocks = lapply(codes, get_prices, start, end, source = 'yahoo')
names(all_stocks) = codes
missing_codes  <- as.character(codes[unlist(lapply(all_stocks, is.null),use.names=FALSE)])

#add from google
all_stocks[missing_codes] <- lapply(missing_codes, get_prices, start, end, source = 'google')
missing_codes  <- as.character(codes[unlist(lapply(all_stocks, is.null),use.names=FALSE)])

#export
stocks = do.call(cbind, all_stocks)
stocks = as.data.frame(stocks)

all_stocks[['FNFV']]
write.csv(stocks, file = 'stocks_data_year.csv', row.names = T, dec = '.')
  