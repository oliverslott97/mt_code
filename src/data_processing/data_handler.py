import dotenv
import sys
import os
from fredapi import Fred
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pytrends.request import TrendReq
import re
import numpy as np
import requests
import functools
import bs4
from random import uniform
from time import sleep
import statsmodels.api as sm
from pandas.tseries.offsets import Week, MonthEnd
import warnings
import pathlib

if os.path.dirname(os.path.dirname(os.getcwd())) not in sys.path:
    sys.path.insert(
        0, 
        os.path.dirname(os.path.dirname(os.getcwd()))
    )
from src.utilities.utilities import *
from src.models.leisen_reimer import IV_solver

dotenv.load_dotenv()
api_keys = {k.replace('API_KEY_', '').lower(): v for k, v in os.environ.items() if k.startswith('API_KEY_')}
clients = {
    'FRED':Fred(api_keys['fred'])
}

_def_start = datetime.today()-timedelta(days=365.24*17)
_sector_index_tickers = {
    'Communication Services':'VOX',
    'Consumer Discretionary':'VCR',
    'Consumer Staples':'VDC',
    'Energy':'VDE',
    'Financials':'VFH',
    'Health Care':'VHT',
    'Industrials':'VIS',
    'Information Technology':'VGT',
    'Materials':'VAW',
    'Real Estate':'VNQ',
    'Utilities':'VPU'
}
_fred_macro_variables = {
    'DTB6':('6 month t-bill', 'US6M'),
    'DTB3':('3 month t-bill', 'US3M'),
    'DTB4WK':('1 month t-bill', 'US1M'),
    'VIXCLS':('CBOE Vol. Index (VIX)', 'VIX'),
    'USEPUINDXD':('Econ. Policy uncertainty index', 'EPU'),
    'BAMLC0A1CAAA':('ICE BofA AAA US corporate index option-adjusted spread', 'ICE_BofA_AAA_SPR'),
    'BAMLH0A0HYM2':('ICE BofA US high yield index option-adjusted spread', 'ICE_BofA_HY_SPR')
}

class RVnMacro:
    
    raw_data_path = test_make_dir(os.path.dirname(os.getcwd()) + '/data/raw')
    processed_data_path = test_make_dir(os.path.dirname(os.getcwd()) + '/data/processed')
    sp500_meta_data_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sector_index_tickers = _sector_index_tickers
    fred_macro_variables = _fred_macro_variables
    news_idx_url = 'https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx?20240105'
    ads_idx_url = 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/ads/ads_index_most_current_vintage.xlsx?la=en&hash=6DF4E54DFAE3EDC347F80A80142338E7'
    
    def __repr__(self):
        return ""
    
    def __init__(self, start=_def_start, end=datetime.today(), format=None):
        self.start = convert_date(start, datetime, format)
        self.end = convert_date(end, datetime, format)
        
    def _print_progress_bar(self, iteration, total, bar_width=48):
        filled_width = int(bar_width * iteration / total)
        empty_width = bar_width - filled_width
        percentage = int(100 * iteration / total)
        bar = '*' * filled_width + ' ' * empty_width
        percentage_pos = (bar_width // 2) - len("{:3d}%".format(percentage)) // 2
        bar_with_percentage = bar[:percentage_pos] + "{:3d}%".format(percentage) \
            + bar[percentage_pos + len("{:3d}%".format(percentage)):]
        sys.stdout.write('\r')
        sys.stdout.write("[{}] {:3d} of {:0d} completed".format(bar_with_percentage, iteration, total))
        sys.stdout.flush()
        
    def _dl_tickers(self, url):
        try:
            df = pd.read_html(url)[0].iloc[:,[0, 1, 2, 6]].rename(
                 columns={'Symbol':'Ticker', 'Security':'Name', 'Date added':'Date'}
            ).assign(
                Ticker=lambda x: x['Ticker'].str.replace('.', '-', regex=False)
            )
            df['Equity Type'] = 'Stock'
            
            return df
        except Exception as e:
            raise Exception(f"Failed to download tickers from {url}: {e}")
        
    def _query_tickers_meta(self, tickers):
        print('Downloading meta data for all tickers. This may take a while...')
        df = pd.DataFrame(columns=['Ticker', 'Market Cap.', 'Industry', 'Biz. Description'])
        
        for i, ticker in enumerate(tickers, start=1):
            self._print_progress_bar(i, len(tickers))
            
            info = yf.Ticker(ticker).info
            r = {
                'Ticker': ticker,
                'Market Cap.': info.get('marketCap', None),
                'Industry': info.get('industry', None),
                'Biz. Description': info.get('longBusinessSummary', None)
            }
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
            
        return df
    
    def _query_tickers_google_trends(self, meta, target_sectors):
        temp = meta[(meta['GICS Sector'].isin(target_sectors)) & (meta['Equity Type'] == 'Stock')]
        print('\n\nDownloading google trends for specified sector. This may take a while...')
        period = convert_date(self.end - timedelta(days=365.24*5), str, '%Y-%m-%d') \
            + ' ' + convert_date(self.end, str, '%Y-%m-%d')
        
        df = pd.DataFrame({'Ticker':[], 'Trend':[]}).astype({'Ticker':'object', 'Trend':'float'})
        for i, (ticker, name) in enumerate(zip(temp.Ticker, temp.Name), start=1):
            self._print_progress_bar(i, temp.shape[0])
            
            name_mod = re.sub(r' \(.*?\)', '', name).replace(' Inc.', ''). replace(',', '') + ' Stock'
            trend_obj = TrendReq(hl='en-US', tz=360)
            try:
                trend_obj.build_payload(
                    kw_list=[str(name_mod)], 
                    timeframe=[period], 
                    cat='7'
                )
                mean_trend = trend_obj.interest_over_time().mean()[str(name_mod)]
            except:
                mean_trend = None
            
            r = {'Ticker': ticker, 'Trend': mean_trend}
            r_df = pd.DataFrame([r])[['Ticker', 'Trend']].astype({'Ticker':'object', 'Trend':'float'})
            df = pd.concat([df, r_df], ignore_index=True)
            
        df = pd.merge(temp, df, on='Ticker', how='inner').sort_values(
            by=['Trend', 'Market Cap.'], ascending=[False, False]
        )
        df = df[~df['Name'].str.contains('(Class A)', regex=False)].iloc[:55].reset_index(drop=True)
        
        return df
    
    def _get_dates_list(self, ticker):
        url = 'https://www.sec.gov/cgi-bin/browse-edgar?type=10-&dateb=&owner=include&count=100&action=getcompany&CIK=%s' % ticker
        headerInfo={'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url,headers=headerInfo)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        noMatch = soup.select('p > center > h1')
        trElems = soup.select('tr')
        '''Regex to get earnings report dates no earlier than 1/1/2000'''
        dateFind = re.compile(r'2\d{3}-\d{2}-\d{2}')
        if noMatch != []:
            return None
        
        dateList = []
        for tr in trElems:
            tdElems = tr.select('td')
            if len(tdElems) == 5 and dateFind.search(tdElems[3].getText()) != None:
                date = tdElems[3].getText()
                converted = datetime.strptime(date,'%Y-%m-%d')
                dateList.append(converted)
                
        return dateList
    
    def _query_sec_announcement_dates(self):
        print('\nDownloading announcement dates for earnings report from SEC EDGAR. This may take a while...')
        temp = self.meta[self.meta['Equity Type'] == 'Stock']#.iloc[:10]
        series = []
        for i, (ticker, cik) in enumerate(zip(temp.Ticker, temp.CIK.astype(int)), start=1):
            self._print_progress_bar(i, temp.shape[0])
            
            s = pd.Series(self._get_dates_list(str(cik)), name=ticker)
            series.append(s)
            
            sleep(uniform(0, 1.5))

        all_dates_set = set()
        for s in series:
            all_dates_set.update(s)

        all_dates_unique = sorted(all_dates_set)
        df = pd.DataFrame(0, index=all_dates_unique, columns=[s.name for s in series])

        for s in series:
            df.loc[s, s.name] = 1

        return df
    
    def _calc_hv(self, prices, window=30):
        if not isinstance(window, int):
            raise TypeError("window must be an integer")
        
        w = str(window) + 'D'
        
        ho = np.log(prices.High / prices.Open)
        lo = np.log(prices.Low / prices.Open)
        co = np.log(prices.Close / prices.Open)
        
        o_c = np.log(prices.Open / prices.Close.shift(1))
        c_c = np.log(prices.Close / prices.Close.shift(1))
        
        F = 252
        N = prices.Close.rolling(window=w).count()
        k = 0.34/(1.34+(N+1)/(N-1))
        
        sigma_cc = np.sqrt(np.maximum(1, F/(N-1))) * np.sqrt((c_c**2).rolling(window=w).sum())
        sigma_rs = np.sqrt(F/(N)) * np.sqrt((ho*(ho-co) + lo*(lo-co)).rolling(window=w).sum())
        sigma_gk = np.sqrt(F/(N)) * np.sqrt(((0.5*(ho-lo)**2)-(2*np.log(2) - 1)*(co**2)).rolling(window=w).sum())
        sigma_yz = np.sqrt(
            np.sqrt(F/np.maximum(1, N-1)) * ((o_c-o_c.rolling(window=w).mean())**2).rolling(window=w).sum() \
                + k * np.sqrt(F/np.maximum(1, N-1)) * ((co-co.rolling(window=w).mean())**2).rolling(window=w).sum() \
                    + (1-k)*sigma_rs**2
        )
        
        sigma_cc.columns = pd.MultiIndex.from_product([['CC1M HV'], sigma_cc.columns])
        sigma_rs.columns = pd.MultiIndex.from_product([['RS1M HV'], sigma_rs.columns])
        sigma_gk.columns = pd.MultiIndex.from_product([['GK1M HV'], sigma_gk.columns])
        sigma_yz.columns = pd.MultiIndex.from_product([['YZ1M HV'], sigma_yz.columns])
        
        df = pd.concat([sigma_cc, sigma_rs, sigma_yz, sigma_gk], axis=1)
        
        return df
    
    def _query_equity_prices(self):
        print('\n\nDownloading equity data from Yahoo! Finance. This may take a while...')
        df = yf.download(
            tickers = self.meta.Ticker.to_list(),
            start=self.start,
            end=self.end,
            interval='1d',
            group_by='column',
            show_errors=False,
            rounding=True,
            actions=True
        )
        
        df_interpolated = df[
            ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        ].interpolate(method='linear')
        df_bfill = df[['Dividends', 'Stock Splits']].bfill()
        
        hv = self._calc_hv(df_interpolated, window=30)
        
        volume = df_interpolated.Volume
        volume.columns = pd.MultiIndex.from_product([['Volume'], volume.columns])
        
        returns = np.log(df_interpolated['Adj Close']/df_interpolated['Adj Close'].shift(1))
        returns.columns = pd.MultiIndex.from_product([['Return'], returns.columns])
        
        splits = df_bfill['Stock Splits']
        splits.columns = pd.MultiIndex.from_product([['Stock Splits'], splits.columns])
        dividends = df_bfill['Dividends']
        dividends.columns = pd.MultiIndex.from_product([['Dividends'], dividends.columns])
        
        # write daily div. for IV calc.
        dividends['Dividends'].reset_index(drop=False).melt(
            id_vars='Date', 
            var_name='Symbol', 
            value_name='Dividend'
        ).to_csv(self.processed_data_path+'/daily_dividends.csv', index=False)
        
        announcements = self._query_sec_announcement_dates()
        announcements = announcements.reindex(
            announcements.index.union(hv.index), 
            fill_value=0
        ).sort_index().loc[hv.index.min():hv.index.max()]
        announcements.columns = pd.MultiIndex.from_product([['Earnings Report'], announcements.columns])
        
        return hv, volume, announcements, returns, splits, dividends
        
    def initialize_tickers(self, tickers=None, 
                           target_sectors=['Information Technology', 'Communication Services']):
        
        if type(target_sectors) is not list:
            target_sectors = [target_sectors]
            
        required_columns = ['Ticker', 'Name', 'GICS Sector', 'Equity Type']

        if tickers is None:
            print('No tickers specified. Setting default to S&P500 tickers.')
            df = self._dl_tickers(self.sp500_meta_data_url)
            print(
                f'\n{df.shape[0]} tickers available'
                f' as of {datetime.today().date()}.'
            )
        elif not isinstance(tickers, pd.DataFrame):
            print('Provided tickers is not a pandas DataFrame. Setting default to S&P500 tickers.')
            df = self._dl_tickers(self.sp500_meta_data_url)
            print(
                f'\n{df.shape[0]} tickers available'
                f' as of {datetime.today().date()}.'
            )
        else:
            if all(col in tickers.columns for col in required_columns):
                if all(tickers[col].apply(lambda x: isinstance(x, str)).all() for col in required_columns):
                    df = tickers
                    print('Valid tickers DataFrame')
                else:
                    print('DataFrame contains non-string values in required columns')
            else:
                print('DataFrame does not contain all the required columns')
                
        self.meta = pd.concat(
            [
                df, 
                pd.DataFrame(
                    {
                        'Ticker': list(self.sector_index_tickers.values()), 
                        'Name': ['Vanguard ' + s for s in self.sector_index_tickers.keys()], 
                        'GICS Sector': list(self.sector_index_tickers.keys()), 
                        'Equity Type': 'ETF'
                    }
                )
            ]
        )
        self.meta = pd.merge(
            self.meta, 
            self._query_tickers_meta(self.meta.Ticker.unique().tolist()), 
            on='Ticker', 
            how='inner'
        ).sort_values(
            by='Market Cap.', 
            ascending=False
        ).reset_index(drop=True)
        self.target_tickers = self._query_tickers_google_trends(self.meta, target_sectors)
        
        return self
    
    def _query_fred_data(self):
        print('\n\nDownloading data from the US federal reserve economic database (FRED).')
        client = clients['FRED']
        df = None
        for i, id in enumerate(self.fred_macro_variables.keys(), start=1):
            self._print_progress_bar(i, len(self.fred_macro_variables))
            
            s = client.get_series(
                series_id=id,
                observation_start=self.start,
                observation_end=self.end
            ).rename(self.fred_macro_variables[id][1])
            if df is None:
                df = pd.DataFrame(s)
            else:
                df = df.merge(pd.DataFrame(s), left_index=True, right_index=True, how='outer')
                
        return df
    
    def _query_news_index(self):
        df = pd.read_excel(self.news_idx_url, sheet_name=1, index_col='date', parse_dates=['date']).rename(
            columns={'News Sentiment':'NEWS'}
        )
        df.index.name = None
        df = df.loc[self.start:self.end]
        
        return df
    
    def _query_ads_index(self):
        df = pd.read_excel(
            self.ads_idx_url, 
            sheet_name=0, 
            index_col='Unnamed: 0', 
            parse_dates=['Unnamed: 0'], 
            date_format='%Y:%m:%d'
        ).rename(columns={'ADS_Index':'ADS'})
        df.index = pd.to_datetime(df.index, format='%Y:%m:%d')
        df = df.loc[self.start:self.end]
        
        return df
        
    def _query_macro(self):
        fred = self._query_fred_data()
        news = self._query_news_index()
        ads = self._query_ads_index()  
        
        df = functools.reduce(
            lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='left'), 
            [fred, news, ads]
        )
        
        return df
    
    def query_data(self):
        (
            self.hv, 
            self.volume, 
            self.announcement, 
            self.returns,
            self.splits,
            self.dividends
        ) = self._query_equity_prices()
        self.macro = self._query_macro()
        
        return self
    
    def _drop_invalid_columns(self, df):
        invalid_columns = []
        for column in df.columns:
            series = df[column]
            first_non_nan = series.first_valid_index()
            if first_non_nan is not None:
                after_first_non_nan = series.loc[first_non_nan:]
                if after_first_non_nan.isnull().any():
                    invalid_columns.append(column)

        if invalid_columns:
            print("Columns with disallowed NaNs after the initial sequence:", invalid_columns)
            drop_tickers = set([j for (_, j) in invalid_columns])
            print(f'Dropping tickers: {", ".join(drop_tickers)}')
            df = df.loc[:, ~df.columns.get_level_values(1).isin(drop_tickers)]
        else:
            print("No columns have disallowed NaNs after the initial sequence.")
            
        return df
    
    def transform(self, sample_freq='W', vol_measure='yz'):
        if sample_freq not in ['W', 'M']:
            raise ValueError("sample_freq must be 'W' or 'M'")
        if not isinstance(vol_measure, str) or vol_measure.lower() not in ['yz', 'cc', 'gk', 'rs']:
            raise ValueError("vol_measure must be one of the following strings: 'yz', 'cc', 'gk', 'rs'")
        
        if sample_freq == 'W':
            pd_freq = 'W-FRI'
            delta = Week
        else:
            pd_freq = sample_freq
            delta = MonthEnd
            
        measurements = {'yz':'YZ1M HV', 'cc':'CC1M HV', 'rs':'RS1M HV', 'gk':'GK1M HV'}
        vol = self.hv.xs(measurements[vol_measure.lower()], level=0, axis=1).iloc[1:]
        all_avgs = vol[self.meta[self.meta['Equity Type'] == 'Stock'].Ticker.to_list()].mean(axis=1)
        sector_tickers = {
            i:self.meta[
                (self.meta['GICS Sector'] == i) & (self.meta['Equity Type'] == 'Stock')
            ].Ticker.to_list() for i in self.meta['GICS Sector'].unique()
        }
        sector_avgs = {k:vol[v].mean(axis=1) for k, v in sector_tickers.items()}        
        print('\n\nCalculating commonality factor for the stock universe.')
        ticker_dfs = {}
        for i, ticker in enumerate(vol.columns.unique(), start=1):
            self._print_progress_bar(i, vol.shape[1])
            ticker_res = {'Sector Commonality':[], 'Market Commonality':[], 'Date':[]}
            for period in pd.date_range(start=vol.index.min(), end=vol.index.max()+delta(1), freq=pd_freq):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    y = vol[ticker].loc[period+delta(-1):period+delta(0)]
                    x_sector = sector_avgs[
                        self.meta[self.meta['Ticker'] == ticker]['GICS Sector'].iloc[0]
                    ].loc[period+delta(-1):period+delta(0)]
                    x_all = all_avgs.loc[period+delta(-1):period+delta(0)]
                    if y.dropna().shape[0] > 2:
                        rsq_sector = sm.OLS(y, sm.add_constant(x_sector)).fit().rsquared
                        rsq_all = sm.OLS(y, sm.add_constant(x_all)).fit().rsquared
                    else:
                        rsq_sector = np.nan
                        rsq_all = np.nan
                ticker_res['Sector Commonality'].append(rsq_sector)
                ticker_res['Market Commonality'].append(rsq_all)
                ticker_res['Date'].append(period)
            ticker_dfs[ticker] = pd.DataFrame(ticker_res).set_index('Date')
        commonaltity = pd.concat(ticker_dfs, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
        
        vol_freq = vol.resample(pd_freq).last()
        vol_freq.columns = pd.MultiIndex.from_product([['HV'], vol_freq.columns])
        def shift(df, col, lag):
            shift = df[col].shift(lag)
            shift.columns = pd.MultiIndex.from_product([[f'{col} L{lag}'], shift.columns])
            return shift
        vol_freq_lagged = vol_freq
        for lag in (1, 2, 3):
            vol_freq_lagged = pd.concat([vol_freq_lagged, shift(vol_freq, 'HV', lag)], axis=1)
        
        price_related_data = pd.concat(
            [
                commonaltity, 
                vol_freq_lagged.iloc[3:],
                self.volume.resample(pd_freq).sum(),
                self.announcement.resample(pd_freq).sum(),
                self.returns.resample(pd_freq).sum(),
                self.dividends.resample(pd_freq).sum(),
                self.splits.resample(pd_freq).sum()
            ], 
            axis=1
        )[3:-1]
        
        categorial_data = []
        for i in ['GICS Sector', 'Equity Type']:
            temp = pd.DataFrame(
                self.meta[
                    self.meta.Ticker.isin(
                        set(price_related_data.columns.get_level_values(1).unique()) & set(self.meta.Ticker)
                    )
                ][['Ticker', 'GICS Sector', 'Equity Type']].set_index('Ticker')[i]
            ).T.reset_index(drop=True)
            temp.columns = pd.MultiIndex.from_product([[i], temp.columns])
            categorial_data.append(temp)
        categorial_data = pd.concat(
            categorial_data, 
            axis=1
        ).rename(index={0:price_related_data.index.min()}).reindex(price_related_data.index, method='ffill')
        
        macro_related_data = self.macro.resample(pd_freq).mean().ffill()
        macro_related_data.columns = pd.MultiIndex.from_product([['Macro'], macro_related_data.columns])
        
        final = pd.merge(
            pd.merge(
                price_related_data, macro_related_data, left_index=True, right_index=True, how='left'
            ),
            categorial_data,
            left_index=True, right_index=True, how='left'
        )
        
        print('\n\n')
        final = self._drop_invalid_columns(final)
        # final.to_parquet(f'{self.raw_data_path}/all_data_{sample_freq}.parquet')
        
        self.meta.to_csv(f'{self.processed_data_path}/meta_data.csv', index=False)
        self.target_tickers.to_csv(f'{self.processed_data_path}/target_tickers.csv', index=False)

        return final
    

class IV:
    
    drive_path = pathlib.Path('/Volumes/ADATA HV620/option_data')
    raw_data_path = test_make_dir(os.path.dirname(os.getcwd()) + '/data/raw/options_data')
    
    def __init__(self, rates, dividends, target_freq='w'):
        self.rates = rates
        self.dividends = dividends
        self.tickers = list(self.dividends.Symbol.unique())
        self.target_freq = target_freq
        
        
    def _print_progress_bar(self, iteration, total, bar_width=48):
        filled_width = int(bar_width * iteration / total)
        empty_width = bar_width - filled_width
        percentage = int(100 * iteration / total)
        bar = '*' * filled_width + ' ' * empty_width
        percentage_pos = (bar_width // 2) - len("{:3d}%".format(percentage)) // 2
        bar_with_percentage = bar[:percentage_pos] + "{:3d}%".format(percentage) \
            + bar[percentage_pos + len("{:3d}%".format(percentage)):]
        sys.stdout.write('\r')
        sys.stdout.write("[{}] {:3d} of {:0d} completed".format(bar_with_percentage, iteration, total))
        sys.stdout.flush()
        
    def initialize_paths(self):
        groups = {
            year: group.Date.to_list() \
                for year, group in self.rates.groupby(self.rates.Date.dt.year)
        }
        available_years = sorted([
            int(str(f).split('/')[-1]) \
                for f in list(self.drive_path.glob('*/[1-2][0-9][0-9][0-9]'))
        ])
        
        print('\n\nSorting through drive to build query structure.')
        query_files = {}
        data_data_n_sample_date = {}
        for i, (k, v) in enumerate(groups.items(), start=1):
            self._print_progress_bar(i, len(groups))
            
            if k not in available_years:
                continue
            
            folder = self.drive_path / f'liveNoGreeks{k}/{k}'
            all_files = [f for f in folder.glob('*/*.csv') if not f.name.startswith('._')]
            available_dates = list(
                {datetime.strptime(re.compile(r'\d{8}').search(f.name).group(), '%Y%m%d') for f in all_files}
            )
            year_query = []
            data_data_n_sample_date_year = {} # data_date:sample_date
            for attempted_date in v:
                
                if not attempted_date in available_dates:
                    data_date = max((date for date in available_dates if date < attempted_date), default=None)
                else:
                    data_date = attempted_date
                    
                if data_date is None and (k - 1) in available_years:
                    prev_year_folder = self.drive_path / f'liveNoGreeks{k-1}/{k-1}'
                    prev_year_files = [
                        f for f in prev_year_folder.glob('*/*.csv') if not f.name.startswith('._')
                    ]
                    prev_year_dates = set(
                        datetime.strptime(re.compile(r'\d{8}').search(f.name).group(), '%Y%m%d') \
                            for f in prev_year_files
                    )
                    data_date = max((date for date in prev_year_dates if date < attempted_date), default=None)
                
                file_tag = datetime.strftime(data_date, '%Y%m%d')
                relevant_files = [f for f in all_files if file_tag in f.name]
                year_query.extend(relevant_files)
                data_data_n_sample_date_year[data_date] = attempted_date
                
            query_files[k] = year_query
            data_data_n_sample_date.update(data_data_n_sample_date_year)
            
        self.query_schema = query_files
        self.sample_date_map = data_data_n_sample_date
        
        return self
    
    def _choose_rate(self, df):
        conditions = [df['DaysToMaturity'] <= 30, df['DaysToMaturity'] <= 90, df['DaysToMaturity'] > 90]
        choices = [df['US1M'], df['US3M'], df['US6M']]

        return np.select(conditions, choices)
    
    def _adjust_rates(self, df):
        df['DaysToMaturity'] = (df.ExpirationDate - df.DataDate).dt.days
        df['ApplicableRate'] = self._choose_rate(df) / 365.25 * df['DaysToMaturity']
        
        return df.drop(columns=['US1M',	'US3M', 'US6M'])
    
    def load_data(self):
        test_make_dir(f'{self.raw_data_path}/{self.target_freq}')
        print('\n\nLoading and transforming options data for full period. This may take a while.')
        for i, (k, v) in enumerate(self.query_schema.items(), start=1):
            self._print_progress_bar(i, len(self.query_schema))
            
            dfs = []
            for f in v:
                dfs.append(
                    pd.read_csv(f, parse_dates=['DataDate', 'ExpirationDate']).query('Symbol in @self.tickers')
                )
                
            df = pd.concat(dfs)
            df['SampleDate'] = df['DataDate'].map(self.sample_date_map)
            df = pd.merge(
                df, 
                self.rates, 
                left_on='SampleDate', 
                right_on='Date', 
                how='left'
            ).drop(columns='Date')
            df = self._adjust_rates(df)
            
            df.to_csv(f'{self.raw_data_path}/{self.target_freq}/{k}.csv', index=False)
            
        return self
    
    def _local_data_paths(self):
        return list(pathlib.Path(self.raw_data_path).iterdir())
    
    def _calculate_adj_dividends(self, df):
        exp_dates = df[
            ['Symbol', 'ExpirationDate', 'DataDate', 'ApplicableRate', 'DaysToMaturity']
        ].drop_duplicates()
        schema = exp_dates.groupby('Symbol').apply(
            lambda x: list(
                zip(x['DataDate'], x['ExpirationDate'], x['ApplicableRate']/x['DaysToMaturity']*365.25/100)
            )
        ).to_dict()
        calculated_values = []
        for symbol, timedelta in schema.items():
            dividends = self.dividends[self.dividends['Symbol'] == symbol]
            for zero, exp, r in timedelta:
                sub = dividends[(dividends.Date <= exp)&(dividends.Date >= zero)]
                div_days = sub[sub.Dividend != 0]
                if not div_days.empty:
                    distance = ((div_days.Date - zero).dt.days / 365.25).values
                    payouts = np.exp(-r*distance)*div_days.Dividend.values
                    adj_div = round(np.sum(payouts), 4)
                else:
                    distance = np.array([])
                    payouts = np.array([])
                    adj_div = 0
                    
                calculated_values.append(
                    {
                        'Symbol': symbol,
                        'ExpirationDate': exp,
                        'DataDate': zero,
                        'TimeDistance': distance,
                        'PayOuts': payouts,
                        'DiscountedDividend': adj_div
                    }
                )
                
        df = pd.merge(
            pd.DataFrame(calculated_values), 
            df, 
            on=['Symbol', 'ExpirationDate', 'DataDate'], 
            how='left'
        )    
                    
        return df
    
    def _filter_for_lr(self, df):
        step = pd.Timedelta(weeks=4) if self.target_freq == 'w' else pd.DateOffset(months=1)
        target = pd.to_datetime(df.SampleDate) + step
        df['days_until_target'] = np.abs((target - pd.to_datetime(df.ExpirationDate)).dt.days)
        min_targets = df.groupby(
            ['Symbol', 'DataDate', 'StrikePrice']
        )['days_until_target'].min().reset_index().rename(columns={'days_until_target': 'MinTargetDate'})
        df_filtered = pd.merge(
            df, 
            min_targets, 
            how='inner', 
            on=['Symbol', 'DataDate', 'StrikePrice']
        )
        df_filtered = df_filtered[df_filtered.days_until_target == df_filtered.MinTargetDate]
        
        df_filtered['strike_dist'] = df_filtered.StrikePrice - df_filtered.UnderlyingPrice
        
        put = df_filtered[(df_filtered.PutCall == 'put')&(df_filtered.strike_dist > 0)]
        put_filter = put.groupby(
            ['Symbol', 'DataDate']
        )['strike_dist'].min().reset_index().rename(columns={'strike_dist': 'MinStrikeDist'})
        filtered_put = pd.merge(put, put_filter, how='inner', on=['Symbol', 'DataDate'])
        filtered_put = filtered_put[
            filtered_put.strike_dist == filtered_put.MinStrikeDist
        ]
        
        call = df_filtered[(df_filtered.PutCall == 'call')&(df_filtered.strike_dist < 0)]
        call_filter = call.groupby(
            ['Symbol', 'DataDate']
        )['strike_dist'].max().reset_index().rename(columns={'strike_dist': 'MinStrikeDist'})
        filtered_call = pd.merge(call, call_filter, how='inner', on=['Symbol', 'DataDate'])
        filtered_call = filtered_call[
            filtered_call.strike_dist == filtered_call.MinStrikeDist
        ]
        
        put_call = pd.concat([filtered_put, filtered_call], axis=0).drop(
            columns=['days_until_target', 'MinTargetDate', 'strike_dist', 'MinStrikeDist']
        )
        
        return put_call.reset_index(drop=True).sort_values(['Symbol', 'SampleDate', 'PutCall'])
        
    def _leisen_reimer_iv(self, df):
        warnings.filterwarnings('ignore', message='invalid value encountered in scalar multiply')
        warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
        warnings.filterwarnings('ignore', message='divide by zero encountered in log')
        warnings.filterwarnings('ignore', message='invalid value encountered in log')
        warnings.filterwarnings('ignore', message='divide by zero encountered in scalar divide')
        
        df = self._calculate_adj_dividends(self._filter_for_lr(df))
        s = []
        for _, row in df.iterrows():
            AmeEurFlag = 'a'
            AdjFlag = False #True if row['DiscountedDividend'] != 0 else False
            CallPutFlag = 'P' if row['PutCall'] == 'put' else 'C'
            ask_bid = []
            for p in ('AskPrice', 'BidPrice'):
                try:
                    iv = IV_solver(
                        AmeEurFlag, 
                        CallPutFlag, 
                        AdjFlag,
                        row['UnderlyingPrice'] - row['DiscountedDividend'],
                        row['StrikePrice'],
                        row['DaysToMaturity']/365.25,
                        row['ApplicableRate']/row['DaysToMaturity']*365.25/100,
                        row['ApplicableRate']/row['DaysToMaturity']*365.25/100,
                        1,
                        row[p],
                        row['TimeDistance'],
                        row['PayOuts']
                    )
                except:
                    iv = np.nan
                
                ask_bid.append(iv)
            
            s.append(round(sum(ask_bid)/2, 4))
            
        df['LR_IV'] = s
        df = df.groupby(
            ['Symbol', 'SampleDate']
        )[['DaysToMaturity', 'LR_IV']].mean().reset_index().rename(columns={'SampleDate':'Date'})
            
        return df
    
    def _clean_for_vix(self, df):
        df = df.drop(columns=[
            'ExpirationDate', 'SampleDate', 'AskSize', 
            'BidSize', 'LastPrice', 'UnderlyingPrice', 'Symbol'
        ]).set_index(['DataDate', 'DaysToMaturity', 'PutCall', 'StrikePrice']).sort_index()
        
        return df
        
    def _create_options_n_yields(self, df):
        df['Premium'] = (df['AskPrice'] + df['BidPrice']) / 2
        yields = df['ApplicableRate'].droplevel(['PutCall', 'StrikePrice']).drop_duplicates()
        df = df.drop(columns=['ApplicableRate'])
        df = df[~df.index.duplicated(keep=False)]
        yields = yields[~yields.index.duplicated(keep=False)]
        
        return df, yields

    def _calc_bid_ask_avg(self, df):
        df = df[df['BidPrice'] > 0]['Premium'].unstack('PutCall')
        df['PutCallDiff'] = (df['call'] - df['put']).abs()
        df['min'] = df['PutCallDiff'].groupby(
            level = ['DataDate','DaysToMaturity']
        ).transform(lambda x: x == x.min())
        
        return df

    def _calc_forward(self, df, yields):
        df = df[df['min'] == 1].reset_index()
        df = pd.merge(df, yields.reset_index(), how = 'left')
        df['Forward'] = df['PutCallDiff'] * np.exp(df['ApplicableRate'] * df['DaysToMaturity'] / 36500)
        df['Forward'] += df['StrikePrice']
        forward = df.set_index(['DataDate','DaysToMaturity'])[['Forward']]
        
        return forward

    def _calc_atm_strike(self, df, forward):
        left = df.reset_index().set_index(['DataDate','DaysToMaturity'])
        df = pd.merge(left, forward, left_index = True, right_index = True)
        mid_strike = df[
            df['StrikePrice'] < df['Forward']
        ]['StrikePrice'].groupby(level = ['DataDate','DaysToMaturity']).max()
        mid_strike = pd.DataFrame({'Mid Strike' : mid_strike})
        
        return mid_strike

    def _sep_otm_cp(self, df, mid_strike):
        left = df.reset_index().set_index(['DataDate', 'DaysToMaturity']).drop('Premium', axis = 1)
        df = pd.merge(left, mid_strike, left_index = True, right_index = True)
        P = (df['StrikePrice'] <= df['Mid Strike']) & (df['PutCall'] == 'put')
        C = (df['StrikePrice'] >= df['Mid Strike']) & (df['PutCall'] == 'call')
        puts, calls = df[P], df[C]
        
        return puts, calls

    def _rem_cons_zero_bids(self, puts, calls):
        calls = calls.assign(zero_bid=lambda df: (df['BidPrice'] == 0).astype(int))
        calls['zero_bid_accum'] = calls.groupby(level = ['DataDate', 'DaysToMaturity'])['zero_bid'].cumsum()
        puts = puts.groupby(
            level = ['DataDate','DaysToMaturity']
        ).apply(lambda x: x.sort_values(['StrikePrice'], ascending = False))
        puts = puts.assign(zero_bid=lambda df: (df['BidPrice'] == 0).astype(int))
        puts['zero_bid_accum'] = puts.groupby(level = ['DataDate' ,'DaysToMaturity'])['zero_bid'].cumsum()
        
        df = pd.concat([calls, puts]).reset_index()
        df = df[(df['zero_bid_accum'] < 2) & (df['BidPrice'] > 0)]
        df['Premium'] = (df['BidPrice'] + df['AskPrice']) / 2
        df = df.set_index(
            ['DataDate', 'DaysToMaturity', 'PutCall', 'StrikePrice']
        )['Premium'].unstack('PutCall').sort_index()
        
        return df

    def _calc_otm_price(self, df, mid_strike):
        left = df.reset_index().set_index(['DataDate','DaysToMaturity'])
        df = pd.merge(left, mid_strike, left_index = True, right_index = True)
        condition1 = df['StrikePrice'] < df['Mid Strike']
        condition2 = df['StrikePrice'] > df['Mid Strike']
        df['Premium'] = (df['put'] + df['call']) / 2
        df['Premium'].loc[condition1] = df['put'].loc[condition1]
        df['Premium'].loc[condition2] = df['call'].loc[condition2]
        df = df[['StrikePrice', 'Mid Strike', 'Premium']].copy()
        duplicated_indices = df.index.duplicated(keep=False)
        df = df[duplicated_indices]
        
        return df

    def _compute_adjoining_strikes_diff(self, group):
        new = group.copy()
        new.iloc[1:-1] = np.array((group.iloc[2:] - group.iloc[:-2]) / 2)
        new.iloc[0] = group.iloc[1] - group.iloc[0]
        new.iloc[-1] = group.iloc[-1] - group.iloc[-2]
        
        return new

    def _calc_diff_adj_strikes(self, df):
        df['dK'] = df.groupby(
            level = ['DataDate', 'DaysToMaturity']
        )['StrikePrice'].transform(self._compute_adjoining_strikes_diff)
        
        return df

    def _calc_strike_contrib(self, df, yields):
        df = pd.merge(df, yields, left_index = True, right_index = True).reset_index()
        df['sigma2'] = df['dK'] / df['StrikePrice'] ** 2
        df['sigma2'] *= df['Premium'] \
            * np.exp(df['ApplicableRate'] * df['DaysToMaturity'] / 36500)
            
        return df

    def _calc_period_idx(self, df, forward, mid_strike):
        df = df.groupby(['DataDate','DaysToMaturity'])[['sigma2']].sum() * 2
        df['Mid Strike'] = mid_strike
        df['Forward'] = forward
        df['sigma2'] -= (df['Forward'] / df['Mid Strike'] - 1) ** 2
        df['sigma2'] /= df.index.get_level_values(1).astype(float) / 365.25
        df = df[['sigma2']]
        
        return df

    def _calc_interp_idx(self, group, freq):
        days = np.array(group['DaysToMaturity'])
        sigma2 = np.array(group['sigma2'])
        
        if days.min() <= freq:
            T1 = days[days <= freq].max()
        else:
            T1 = days.min()
        
        T2 = days[days > T1]
        if len(T2) > 0:
            T2 = T2.min()
        else:
            T2 = T1
        
        sigma_T1 = sigma2[days == T1][0]
        sigma_T2 = sigma2[days == T2][0]
            
        return pd.DataFrame([{'T1' : T1, 'T2' : T2, 'sigma2_T1' : sigma_T1, 'sigma2_T2' : sigma_T2}])

    def _interp_vix(self, df, freq):
        df = df.copy()
        for t in ['T1','T2']:
            df['days_' + t] = df[t].astype(float) / 365.25
            df[t] = (df[t] - 1) * 1440. + 510 + 930

        df['sigma2_T1'] = df['sigma2_T1'] * df['days_T1'] * (df['T2'] - freq * 1440.)
        df['sigma2_T2'] = df['sigma2_T2'] * df['days_T2'] * (freq * 1440. - df['T1'])
        df['IV_VIX'] = ((df['sigma2_T1'] + df['sigma2_T2']) / (df['T2'] - df['T1']) * 365.25 / freq) ** .5
        
        return df
    
    def _calc_vix(self, df, freq):
        options, yields = self._create_options_n_yields(self._clean_for_vix(df))
        options2 = self._calc_bid_ask_avg(options)
        forward = self._calc_forward(options2, yields)
        mid_strike = self._calc_atm_strike(options2, forward)
        puts, calls = self._sep_otm_cp(options, mid_strike)
        options3 = self._rem_cons_zero_bids(puts, calls)
        options4 = self._calc_diff_adj_strikes(self._calc_otm_price(options3, mid_strike))
        contrib = self._calc_strike_contrib(options4, yields)
        sigma2 = self._calc_period_idx(contrib, forward[~forward.index.duplicated(keep=False)], mid_strike)
        two_sigmas = sigma2.reset_index().groupby('DataDate').apply(
            self._calc_interp_idx, 
            freq
        ).groupby(level = 'DataDate').first()
        vix = self._interp_vix(two_sigmas, freq)

        return round(vix['IV_VIX'], 4)
        
    def _run_vix_calc(self, df, freq=28):
        dfs = []
        for symbol in df.Symbol.unique():
            sub = df[df.Symbol == symbol]
            try:
                vix = pd.DataFrame(self._calc_vix(sub, freq)).reset_index(drop=False)
                vix['Symbol'] = symbol
                dfs.append(vix.reset_index(drop=True))
            except:
                None
            
        return pd.concat(dfs, axis=0)
    
    def _run_calculations(self):
        freq = 28 if self.target_freq == 'w' else 30
        data_paths = list(pathlib.Path(self.raw_data_path).iterdir())
        print('\n\nCalculating IV for all years. This may take a while.')
        for i, path in enumerate(data_paths, start=1):
            self._print_progress_bar(i, len(data_paths))
            df = pd.read_csv(path, parse_dates=['DataDate', 'ExpirationDate', 'SampleDate'])
            lr_path = test_make_dir(self.raw_data_path + '/lr')
            vix_path = test_make_dir(self.raw_data_path + '/vix')
            self._leisen_reimer_iv(df).to_csv(lr_path + f'/{path.name}', index=False)
            self._run_vix_calc(df, freq).to_csv(vix_path + f'/{path.name}', index=False)
            
        return self
    
    def calc_implied_vol(self):
        self._run_calculations()
        
        lr_data = list(pathlib.Path(self.drive_path + '/lr').iterdir())
        vix_data = list(pathlib.Path(self.drive_path + '/vix').iterdir())
        
        lr = pd.concat([pd.read_csv(i, parse_dates=['Date']) for i in lr_data], axis=0)
        vix = pd.concat([pd.read_csv(i, parse_dates=['DataDate']) for i in vix_data], axis=0)
        
        lr['DataDate'] = lr['Date'].map({v: k for k, v in self.sample_date_map.items()})
        
        iv = pd.merge(vix, lr, on=['Symbol', 'DataDate'], how='outer')
        iv['LR_IV'] = iv['LR_IV']*np.sqrt(28/iv['DaysToMaturity'])
        
        iv = iv.sort_values(['Symbol', 'Date'])
        
        iv['LR_IV'] = iv.groupby('Symbol')['LR_IV'].transform(lambda x: x.interpolate())
        iv['IV_VIX'] = iv.groupby('Symbol')['IV_VIX'].transform(lambda x: x.interpolate())
        
        return iv.drop(columns=['DataDate', 'DaysToMaturity']).reset_index(drop=True).rename(
            columns={'LR_IV':'IVLeisenReimer', 'IV_VIX':'IVVIXMethod'}
        )
        