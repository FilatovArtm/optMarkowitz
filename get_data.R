library(quantmod)
library(plyr)

start = '2014-03-31'
end = '2014-06-30'

codes = read.csv('stocks.csv', encoding = 'UTF-8', sep=',')[,2]
getSymbols(Symbols = as.character(codes[3]),from = start, to = end)
dates <- index(get(as.character(codes[3])))
all_stocks_yahoo = data.frame(date = dates)

# Get daily return of the asset 
# get_prices <- function(x, start, end, source){
#   return = c()
#   tryCatch({
#   prices = getSymbols(Symbols = as.character(x),from = start, to = end, 
#                      auto.assign = FALSE, warnings = FALSE, src = source)[,c(1,4)]
#   return = (prices[,2]/prices[,1])-1
#   
#   }, error=function(e){cat("ERROR :",conditionMessage(e), "\n") })
#   return(return)
# }

get_prices <- function(x, start, end, source){
  return = c()
  tryCatch({
  prices = getSymbols(Symbols = as.character(x),from = start, to = end,
                     auto.assign = FALSE, warnings = FALSE, src = source)[,4]
  return = diff(prices)/(prices[-(prices)])

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


write.csv2(stocks, file = 'stocks_data.csv', row.names = T)
  