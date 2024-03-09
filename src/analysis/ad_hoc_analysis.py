# commonality code:
# query_dict = query_obj.meta[query_obj.meta['Equity Type']=='Stock'].groupby('GICS Sector')['Ticker'].apply(list).to_dict()
# query_dict['All'] = query_obj.meta[query_obj.meta['Equity Type']=='Stock'].Ticker.to_list()
# avg_m_sum_stat = []
# for i, (sector, tickers) in enumerate(query_dict.items()):
#     union = list(set(rv_yz_M.columns.get_level_values(1).unique()) & set(tickers))
#     res = rv_yz_M.xs('Return', level=0, axis=1)[union]
#     count = res.shape[1]
#     ret = res.mean(axis=0).mean() * 100
#     skew = res.skew(axis=0).mean()
#     kurt = res.kurt(axis=0).mean()

#     sector = pd.DataFrame(
#         {
#             'Sector':sector,
#             'Number of stocks':count,
#             'Return':ret,
#             'Skewness':skew,
#             'Kurtosis':kurt
#         },
#         index = [i]
#     )
#     avg_m_sum_stat.append(sector)
    
# sum_stat = pd.concat(avg_m_sum_stat)
# display(sum_stat)