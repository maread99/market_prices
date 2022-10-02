"""Get prices via yahooquery."""

from __future__ import annotations

import copy
import functools
import warnings
from typing import Dict, List, Optional, Union

from pandas import DataFrame
import pandas as pd
import numpy as np
import pydantic
import exchange_calendars as xcals
import yahooquery as yq

from market_prices import errors, helpers, intervals, mptypes
from market_prices.prices import base

from ..mptypes import Calendar, Symbols
from .config import config_yahoo


class PricesYahoo(base.PricesBase):
    """Retrieve and serve price data sourced via yahooquery.

    Users of this class should see the disclaimer under the 'Notes' section
    further below.

    Parameters
    ----------
    symbols: Union[str, List[str]]
        Symbols for which require price data. For example:
            'AMZN'
            'FB AAPL AMZN NFLX GOOG MSFT'
            ['FB', 'AAPL', 'AMZN']

    calendars :
    Optional[
        Union[
            mptypes.Calendar,
            list[mptypes.Calendar],
            dict[str, mptypes.Calendar],
        ]
    ], default: evaluated
        Calendar(s) defining trading times and timezones for `symbols`.

        By default a calendar for each symbol is ascertained by
        inspection and mapping of various fields available via Yahoo.
        A `errors.CalendarError` is raised in the event that a calendar
        cannot be ascertained for every symbol. In this case the client
        should pass `calendars` to define a calendar for at least all
        symbols for which a calendar cannot be otherwise ascertained.
        Also, `calendars` can be passed to override any or all the
        default calendars that would otherwise be assigned.

        A single calendar representing all `symbols` can be passed as
        an mptype.Calendar, specifically any of:
            Instance of a subclass of
            `exchange_calendars.ExchangeCalendar`. Calendar 'side' must
            be "left".

            `str` of ISO Code of an exchange for which the
            `exchange_calendars` package maintains a calendar. See
            https://github.com/gerrymanoim/exchange_calendars#calendars
            or call market_prices.get_exchange_info`.

            `str` of any other calendar name supported by
            `exchange_calendars`, as returned by
            `exchange_calendars.get_calendar_names`

        Multiple calendars, each representing one or more symbols, can
        be passed as any of:
            List of mptypes.Calendar (i.e. defined as for a single
            calendar). List should have same length as `symbols` with each
            element relating to the symbol at the corresponding index.

            Dictionary with items representing only those symbols for which
            wish to define a calendar. Any symbol not included to keys will
            be assigned, if possible, the default calendar assigned for the
            symbol.
                key: str
                    symbol.
                value: mptypes.Calendar (i.e. as for a single calendar)
                    Calendar corresponding with symbol.

        All calendars should have a first session no later than
        `base_limits` for the daily interval.

    lead_symbol
        Symbol with calendar that should be used as the default
        calendar to evaluate period from period parameters. If not
        passed default calendar will be defined as the most common
        calendar.

    delays : default: evaluated
        Real-time price delay for each symbol, in minutes.

        By default, `PricesYahoo` will attempt to evaluate the price
        delay associated with each symbol. Note: the Yahoo API does not
        make a field available that reliably reflects the price delay
        for all symbols ('exchangeDataDelayedBy' is unreliable).
        `PricesYahoo` uses a combination of interrogation of a Yahoo
        field that indicates real-time prices and hard-coded mappings
        (both specific and generalised). The following assumptions
        should be noted (delays in minutes):
            Cryptocurrencies: 0
            Currencies: 0
            Symbols that include '=F': 10

        A `ValueError` is raised if a delay cannot be evaluated for
        a symbol. In this case client should pass `delays` to define
        the delays for at least those symbols that cannot otherwise be
        evaluated. Also, `delays` can be passed to override any or all
        of the evaluated delays that would otherwise be assigned.
        Alternatively a default delay for any symbol can be set by
        including the symbol to the `DELAY_MAPPING` dictionary
        maintained on `config_yahoo.py`.

        For a single symbol, pass an `int` representing the delay in
        minutes, or 0 for real-time.

        For multiple symbols that do not all have the same delay pass
        as either:
            List of int of same length as symbols where each item
            relates to symbol at corresponding index.

            Dictionary with items representing only those symbols for
            which wish to define a delay. Any symbol not included
            to keys will be assigned the default evaluated delay.
                key: str
                    symbol.
                value: int
                    Delay corresponding with symbol.

        Note: An inaccurately evaluated delay will have the following
        effects:
            If defined delay is less than the actual delay then prices
            may become 'fixed' over the last indices of requested
            data. Any fixed data will not refresh on further requests.

            If defined delay is greater than the actual delay then
            requested data will not be available up to the timestamp to
            which it would be expected to be available.

    Notes
    -----
    --DISCLAIMER--
    `market_prices` is NOT in any way affiliated, partnered, sponsored or
    endorsed by Yahoo. Users of this class should make enquiries to satisfy
    themselves that they are eligible to receive data from 'Yahoo APIs' and
    are in compliance with the license requirements and Terms of Service
    under which the 'Yahoo APIs' may be accessed, to include restrictions
    concerning NO COMMERCIAL USE. See the `market_prices` home page for
    references.

    This following sections are likely to be of interest only to
    developers.

    --yahooquery data--
    Price data is returned as a DataFrame indexed with a
    pd.DatatimeIndex if interval is intraday or pd.Index of Type object
    if interval daily.

    The indices define the left value of the interval that the row
    covers. For example, the 10.20 indice of 5 minute interval data
    will offer the high, low, open, and close values for the period
    10.20 - 10.25, inclusive of the 10.20 minute and exclusive of the
    10.25 minute.

    For daily prices:
        In theory prices are indexed with the corresponding session.
        In reality they are indexed with the date component of a
        timestamp that represents the session open either as UTC or
        UTC plus the GMT offset (depending on whether adj_timezone
        passed as False or True respectively). Consequently if
        `adj_timezone` is False and the UTC open is before midnight
        (for example ASX opens as 23.00 UTC) then the index is a day
        too early. This bug is fixed by localizing some
        yahooquery methods and making changes to the localized methods
        `_yq_history_dataframe` and `_yq_events_to_dataframe`.

        Separately, for a time following a session close the yahoo API
        represents the final session with two timestamps - one stamped
        as the session open (the normal way in which the API stamps a
        session) and another as the 'live indice' stamped with,
        presumedly, the time of the last trade. The two indices are
        present in the yahooquery data. The duplication is passed
        through and handled within `_resolve_current_ts_daily`.

    Indices will be UTC or local depending on interval and
    whether 'adj_timezone' is passed to `yq.Ticker.history`
    method as False or True (default):

        For daily intervals:
            indices represent session (i.e. no time component). For
            example the indice 2021-03-22 could represent the CME
            session that opened 2021-03-21 17.00 and closed
            2021-03-22 16.00.

        For intraday intervals:
            If 'adj_timezone' is passed as False, all times are
            returned as UTC times.

            If 'adj_timezone' is passed as True then yahooquery applies
            an offset against the UTC times to return times as local.
            NB, this offset is constant and does not account for DST.
            Consequently during periods where an exchange observes DST
            the indices will be an earlier than true local time that
            is observing DST. Although of no consequence for this class
            (ticker.history is only called with adj_timezone as False)
            a fix is included to `_yq_history_dataframe` is
            anticipation of offering PR to fix yahooquery source.

    Importantly, when datetimes are passed as the 'start' and/or 'end'
    parameters of yq.Ticker.history, these datetimes are assumed as UTC
    times regardless of the value of any passed 'adj_timezone'
    parameter.

    Is the 'end' parameter datetime included in the data?
        If the interval is intraday, No. For example, if request data
        at 5 minute intervals ending 15:04 on a particular day, the
        final index will be 14.55 and will represent the interval
        14.55 to 15.00. However, pass end as 15.05 and final index
        will be 15.00, covering period from 15.00 to 15.05.

        If the interval is daily, yes if the time componenent is later
        than the first trading time of that session (based on 'utc'),
        otherwise no.

    During market hours, when the interval is intraday the most recent
    timestamp will be a current timestamp representing a snapshot at
    the time the data was sent. The open, high, low and close of this
    current snapshot will all have the same value and volume will be
    0. NB when an interval completes in real time, for example when
    passing 10.25, a timestamp for the 'open' interval, i.e. 10.25 in
    this case, will not appear immediately but rather, typically, a
    minute or so later. So, at 10.26 the last indice would be the
    current snapshot at 10.26 and the one before might be 10.20. Then
    by 10.27 the timestamp for the most recent interval may have
    initialised, such that the last indice would be the current
    snapshot at 10.27 and the one before would be 10.25 representing
    the open interval. The values for this open interval will update
    if the data is re-requested before the end of the interval.

    During market hours, when the interval is daily the most recent
    timestamp will have indice as the live session (Index is of type
    object) and values will reflect prices as at the time the data was
    requested.

    Volume glitch
        The API seems to have a nasty glitch when requesting
        intraday volume data. The volume for the first indice of
        requested data for each symbol will either by 0 or understate
        the true value.

        The effect can be produced simply by requesting intraday data
        over a period and comparing it to data requested to start one
        indice prior or subsequent. This shows that accurate volume
        data for the first indice is available, it's just that
        it doesn't return correctly. There is one exception - for some
        intervals the volume for the first indice of every session is
        0.

        A fix is applied within `_request_yahoo` to accurately return
        volumen data when it's otherwise available.

    -- Reindexing --
    PricesYahoo reindexes intra-day price data to introduce indices for
    missing data. Introduced indices are then appropriately filled with
    data.

    Why Reindex?
    Yahoo returns intraday prices with gaps (missing indices) for any
    period where no price data was recorded. So, for example, for a certain
    day '5m' data could have sequential indices:
        10.35, 10.40, 10.50, 11.10
    i.e. if no price was registered between 10.45 and 10.50, or 10.55 and
    11.10 then no indice will be included to represent these missing
    datapoints, with the consequence that the period between one data point
    and the next is not constant (the lesser traded the instrument the
    greater the effect).

    Reindexing fills in the gaps. This complies with the PricesBase
    implementation's requirement that all indices are complete (no gaps)
    during trading times over requested period.
    """

    BaseInterval = intervals._BaseInterval(  # pylint: disable=protected-access
        "BaseInterval",
        dict(
            T1=intervals.TIMEDELTA_ARGS["T1"],
            T2=intervals.TIMEDELTA_ARGS["T2"],
            T5=intervals.TIMEDELTA_ARGS["T5"],
            H1=intervals.TIMEDELTA_ARGS["H1"],
            D1=intervals.TIMEDELTA_ARGS["D1"],
        ),
    )

    # for 2 minute interval yahooquery accepts requests up to 60 days,
    # although data only available for last 43 days (might even be less).
    BASE_LIMITS = {
        BaseInterval.T1: pd.Timedelta(30, "D"),
        BaseInterval.T2: pd.Timedelta(43, "D"),
        BaseInterval.T5: pd.Timedelta(60, "D"),
        BaseInterval.H1: pd.Timedelta(730, "D"),
        BaseInterval.D1: None,
    }

    YAHOO_EXCHANGE_TO_CALENDAR = config_yahoo.EXCHANGE_TO_CALENDAR
    YAHOO_DELAY_MAPPING = config_yahoo.DELAY_MAPPING

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        symbols: Union[str, List[str]],
        calendars: Optional[mptypes.Calendars] = None,
        lead_symbol: Optional[str] = None,
        delays: Optional[Union[int, List[int], Dict[str, int]]] = None,
    ):
        symbols = helpers.symbols_to_list(symbols)
        self._ticker = yq.Ticker(
            symbols,
            formatted=False,
            asynchronous=True,
            max_workers=8,
            proxies=None,
            validate=True,
        )

        if self._ticker.invalid_symbols is not None:
            raise ValueError(
                "The following symbols are not recognised by the yahoo API:"
                f" {self._ticker.invalid_symbols}."
            )

        self._yahoo_exchange_name: dict[str, str]
        self._set_yahoo_exchange_name()

        if calendars is None or (
            isinstance(calendars, dict) and len(calendars) < len(symbols)
        ):
            calendars = self._ascertain_calendars(calendars)

        if delays is None or (isinstance(delays, dict) and len(delays) < len(symbols)):
            delays = self._ascertain_delays(delays)

        self._cache_vol_bug_adj_start: None | tuple[pd.Timestamp, pd.Timestamp] = None
        self._set_daily_bi_limit()
        super().__init__(symbols, calendars, lead_symbol, delays)

    # Methods called via constructor

    def _set_yahoo_exchange_name(self):
        d = {}
        for s in self._ticker.symbols:
            d[s] = self._ticker.quotes[s]["fullExchangeName"]
        self._yahoo_exchange_name = d

    def _ascertain_calendars(
        self, calendars: dict[str, Calendar] | None
    ) -> dict[str, Calendar]:
        """Return dict of calendars for all symbols.

        All items of `calendars` will be included to return.
        """
        if calendars is None:
            calendars = {}
        all_names = xcals.get_calendar_names()
        for s in self._ticker.symbols:
            if s in calendars:
                continue
            exchange = self._yahoo_exchange_name[s]
            if exchange in all_names:
                cal = exchange
            elif exchange in self.YAHOO_EXCHANGE_TO_CALENDAR:
                cal = self.YAHOO_EXCHANGE_TO_CALENDAR[exchange]
            elif self._ticker.quotes[s]["market"] == "us24_market":
                cal = "us_futures"
            else:
                msg = f"Unable to ascertain calendar for symbol '{s}'."
                raise errors.CalendarError(msg)
            calendars[s] = cal
        return calendars

    def _yahoo_symbol_mkt_code(self, s) -> str | None:
        """Return market code for a given symbol.

        For example, where s='AZN.ST', -> '.ST'.
        """
        if "." not in s:
            return None
        code = s[s.find(".") :]
        if code.count(".") > 1:
            return self._yahoo_symbol_mkt_code(code)
        return code

    def _delay_mapping(self, symbol: str) -> int | None:
        if symbol in self.YAHOO_DELAY_MAPPING:
            return self.YAHOO_DELAY_MAPPING[symbol]
        mkt_code = self._yahoo_symbol_mkt_code(symbol)
        if mkt_code is not None and mkt_code in self.YAHOO_DELAY_MAPPING:
            return self.YAHOO_DELAY_MAPPING[mkt_code]
        else:
            return None

    @functools.cached_property
    def _real_time(self) -> dict[str, bool]:
        """Return dictionary indicating if symbols have real time pricing."""
        d = {}
        price = self._ticker.price
        for s in self._ticker.symbols:
            info = price[s]
            d[s] = info["regularMarketSource"] == "FREE_REALTIME"
            if not d[s]:
                d[s] = info["exchange"] == "NYQ" and info["quoteType"] == "EQUITY"
        return d

    def _ascertain_delays(self, delays: dict[str, int] | None) -> dict[str, int]:
        """Return dict of delays for all symbols.

        All items of `delays` will be included to return.
        """
        d = delays if delays is not None else {}
        for s in self._ticker.symbols:
            if s in d:
                continue
            delay = None
            delay_mapping = self._delay_mapping(s)
            if delay_mapping is not None:
                delay = delay_mapping
            elif self._real_time[s]:
                delay = 0
            elif "=F" in s:
                delay = 10
            elif self._yahoo_exchange_name[s] in ["CCC", "CCY"]:
                delay = 0
            else:
                msg = (
                    f"Unable to evaluate price delay for symbol {s}."
                    " Pass `delays` to constructor as a dict including at"
                    f" least item {s}: int  where int represents price delay"
                    f" for {s} in minutes, for example {{{s}: 15}}."
                )
                raise ValueError(msg)
            d[s] = delay
        return d

    @functools.cached_property
    def _first_trade_dates(self) -> dict[str, pd.Timestamp | None]:
        quote_type = self._ticker.quote_type
        first_trade_dates: dict[str, pd.Timestamp | None] = {}
        for s in self._ticker.symbols:
            first_trade_str = quote_type[s]["firstTradeDateEpochUtc"]
            if first_trade_str is None:
                first_trade_dates[s] = None
            else:
                first_trade_dates[s] = pd.Timestamp(first_trade_str).normalize()
        return first_trade_dates

    def _set_daily_bi_limit(self):
        dates = [date for date in self._first_trade_dates.values() if date is not None]
        if dates:
            earliest = min(dates)
            self._update_base_limits({self.BaseInterval.D1: earliest})

    # Localised yahooquery methods.

    @staticmethod
    def _yq_events_to_dataframe(data, symbol) -> pd.DataFrame:
        # yahooquery method localized and amended
        dataframes = []
        for event in ["dividends", "splits"]:
            try:  # pylint: disable=too-many-try-statements
                df = pd.DataFrame(data[symbol]["events"][event].values())
                df.set_index("date", inplace=True)
                df.index = pd.to_datetime(df.index, unit="s")
                df.index = df.index.tz_localize("UTC")
                if event == "dividends":
                    df.rename(columns={"amount": "dividends"}, inplace=True)
                else:
                    df["splits"] = df["numerator"] / df["denominator"]
                    df = df[["splits"]]
                dataframes.append(df)
            except KeyError:
                pass
        if len(dataframes) > 1:
            return pd.merge(
                dataframes[0],
                dataframes[1],
                how="outer",
                left_index=True,
                right_index=True,
            )
        else:
            return dataframes[0]

    def _yq_history_dataframe(
        self, data, symbol, params, adj_timezone=True
    ) -> pd.DataFrame:
        # yahooquery localized and amended method to fix assumption that date
        # component of index always reflects session.
        s = symbol

        df = pd.DataFrame(data[s]["indicators"]["quote"][0])
        if data[s]["indicators"].get("adjclose"):
            df["adjclose"] = data[s]["indicators"]["adjclose"][0]["adjclose"]
        df.index = pd.to_datetime(data[s]["timestamp"], unit="s")
        df.index = df.index.tz_localize("UTC")

        df.dropna(inplace=True)
        if data[s].get("events"):
            df = pd.merge(
                df,
                self._yq_events_to_dataframe(data, s),
                how="outer",
                left_index=True,
                right_index=True,
            )

        daily = params["interval"][-1] not in ["m", "h"]
        if adj_timezone or daily:
            # df.index = df.index.date  # OLD yahooquery bug - date component of
            # index does not always reflect session.
            tz = data[s]["meta"]["exchangeTimezoneName"]
            if not daily:
                df.index = df.index.tz_convert(tz)
            else:
                last_trade_secs = data[s]["meta"]["regularMarketTime"] * 10**9
                last_t = pd.Timestamp(last_trade_secs, tz="UTC")
                live_indice = df.index[-1] >= last_t - pd.Timedelta(2, "S")
                slce = slice(-1) if live_indice else slice(None)
                idx = df.index[slce].tz_convert(tz)  # to local open time
                idx = idx.round("d")  # round from local open time
                idx = idx.tz_localize(None)
                if live_indice:
                    last_indice = df.index[-1]
                    if adj_timezone:
                        last_indice = last_indice.tz_convert(tz)
                    last_indice = last_indice.tz_localize(None)
                    idx = idx.insert(len(idx), last_indice)
                df.index = idx
        return df

    def _yq_historical_data_to_dataframe(
        self, data, params, adj_timezone
    ) -> dict | pd.DataFrame:
        # localized yahooquery method. Unchanged except rerouting calls to
        # other localized methods.
        d = {}
        for symbol in self._ticker._symbols:  # pylint: disable=protected-access
            if "timestamp" in data[symbol]:
                d[symbol] = self._yq_history_dataframe(
                    data, symbol, params, adj_timezone
                )
            else:
                d[symbol] = data[symbol]
        if all(isinstance(v, pd.DataFrame) for v in d.values()):
            df = pd.concat(d, names=["symbol", "date"], sort=False)
            if "dividends" in df.columns:
                df["dividends"].fillna(0, inplace=True)
            if "splits" in df.columns:
                df["splits"].fillna(0, inplace=True)
            return df
        return d

    def _yq_history(  # pylint: disable=too-many-arguments
        self,
        period="ytd",
        interval="1d",
        start=None,
        end=None,
        adj_timezone=True,
        adj_ohlc=False,
    ) -> pd.DataFrame | dict:
        """Get historical pricing data.

        Pulls historical pricing data for a given symbol(s)

        Parameters
        ----------
        period: str, default ytd, optional
            Length of time
        interval: str, default 1d, optional
            Time between data points
        start: str or datetime.datetime, default None, optional
            Specify a starting point to pull data from.  Can be expressed as a
            string with the format YYYY-MM-DD or as a datetime object
        end: str of datetime.datetime, default None, optional
            Specify a ending point to pull data from.  Can be expressed as a
            string with the format YYYY-MM-DD or as a datetime object.
        adj_timezone: bool, default True, optional
            Specify whether or not to apply the GMT offset to the timestamp
            received from the API.  If True, the datetimeindex will be adjusted
            to the specified ticker's timezone.
                If interval is '1d' or higher then adj_timezone applies only
                    to the last timestamp and only if that last timestamp
                    represents a live session (i.e. a day that has yet to
                    close).
        adj_ohlc: bool, default False, optional
            Calculates an adjusted open, high, low and close prices according
            to split and dividend information

        Returns
        -------
        pandas.DataFrame
            historical pricing data. If interval is '1d' or higher then
                indices are dates representing closed sessions. An
                exception is if the period runs to 'now' in which case the
                final indice will reflect the current session as at 'now'.
        """
        # localized yahooquery method. Unchanged except rerouting calls to
        # other localized methods.
        # pylint: disable=protected-access, redefined-outer-name
        _convert_to_timestamp = yq.ticker._convert_to_timestamp

        config = self._ticker._CONFIG["chart"]
        intervals = config["query"]["interval"]["options"]
        if start or period is None or period.lower() == "max":
            start = _convert_to_timestamp(start)
            end = _convert_to_timestamp(end, start=False)
            params = {"period1": start, "period2": end}
        else:
            params = {"range": period.lower()}
        if interval not in intervals:
            values = ", ".join(intervals)
            msg = f"Interval values must be one of {values}"
            raise ValueError(msg)
        params["interval"] = interval.lower()
        if params["interval"] == "1m" and period == "1mo":
            df = self._ticker._history_1m(adj_timezone, adj_ohlc)
        else:
            data = self._ticker._get_data("chart", params)
            df = self._yq_historical_data_to_dataframe(data, params, adj_timezone)
        if adj_ohlc and "adjclose" in df:
            df = self._ticker._adjust_ohlc(df)
        return df

    # Methods to request data from yahooquery.

    @staticmethod
    def _bi_to_source_key(interval: intervals.BI) -> str:
        """Map interval to value for source's interval parameter."""
        mapping = {"MS": "mo", "T": "m"}
        if interval.freq_unit in mapping:
            return str(interval.freq_value) + mapping[interval.freq_unit]
        else:
            return interval.as_pdfreq.lower()  # as yahooquery value

    @staticmethod
    def _prices_for_some_symbols(d: dict) -> pd.DataFrame | None:
        """Create Dataframe from dictionary returned from `_ticker.history`.

        Create DataFrame from `d` returned by _ticker.history only if
        prices are available for at least one symbol. Symbols for which
        prices are not available will be represented with missing values.

        Returns None if prices unavailable for all symbols.
        """
        dfs, non_dfs = {}, []
        for k, v in d.items():
            if isinstance(v, DataFrame):
                dfs[k] = v
            else:
                non_dfs.append(k)

        if not dfs:
            return None

        no_values = list(dfs.values())[0].copy()
        for col in no_values:
            val = 0 if col == "volume" else np.NaN
            no_values[col].values[:] = val

        for s in non_dfs:
            dfs[s] = no_values.copy()

        prices = pd.concat(dfs)
        prices.index.set_names(["symbol", "date"], inplace=True)
        return prices

    def _adj_start_for_vol_glitch(
        self, start: pd.Timestamp, interval: intervals.BI
    ) -> pd.Timestamp:
        if interval == self.BaseInterval.T1:
            # simple cache to not repeat evaluation twice for each loop of
            # `_request_data_1m`.
            cache = self._cache_vol_bug_adj_start
            if cache is not None and cache[0] == start:
                return cache[1]
        prior_indices_left = []
        for cal in self.calendars_unique:
            ts = cal.minute_to_trading_minute(start, "previous")
            prior_indices_left.append(cal.minute_offset(ts, -interval.as_minutes))
        adj_start = max(min(prior_indices_left), self._pdata[interval].ll)
        if interval == self.BaseInterval.T1:
            self._cache_vol_bug_adj_start = (start, adj_start)
        return adj_start

    def _request_yahoo(
        self,
        interval: intervals.BI,
        start: pd.Timestamp | None,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Get price data from Yahoo.

        Parameters
        ----------
        interval
            Time delta covered by each row.

        start
            Range start date.

        end
            Range end date.

        Raises
        ------
        errors.PricesUnavailableFromSourceError
            If call to yq.Ticker.history fails to return prices.

        Notes
        -----
        See PricesYahoo class documentation for notes on internals.
        """
        volume_glitch = interval.is_intraday
        if volume_glitch:
            assert start is not None
            adj_start = self._adj_start_for_vol_glitch(start, interval)
            if interval == self.BaseInterval.T1:
                if start - adj_start >= pd.Timedelta(6, "D"):
                    # bail on fix to avoid not getting any data.
                    start_: pd.Timestamp | None = start
                    volume_glitch = False
            start_ = adj_start

        else:
            start_ = start

        # "ytd" is default, ignored if start and end both passed
        period = "max" if start_ is None else "ytd"
        interval_ = self._bi_to_source_key(interval)
        prices: pd.DataFrame | dict | None
        prices = self._yq_history(
            period, interval=interval_, start=start_, end=end, adj_timezone=False
        )

        def raise_error():
            params = {
                "interval": interval_,
                "start": start,
                "end": end,
            }
            raise errors.PricesUnavailableFromSourceError(params, prices)

        if isinstance(prices, dict):
            prices = self._prices_for_some_symbols(prices)
            if prices is None:
                raise_error()

        assert isinstance(prices, pd.DataFrame)
        if volume_glitch:
            drop_indices = prices[prices.index.droplevel(0) < start].index
            prices = prices.drop(drop_indices)
            if prices.empty:
                raise_error()

        return prices

    # Methods to tidy data returned by yahooquery

    @staticmethod
    def _fill_reindexed(
        df: pd.DataFrame,
        calendar: xcals.ExchangeCalendar,
        bi: intervals.BI,
        symbol: str,
    ) -> pd.DataFrame:
        """Fill, for a single `symbol`, rows with missing data."""
        # fill index grouped by day or session to avoid filling a session's
        # initial values with prior session's close value.
        # pylint: disable=too-many-locals
        na_rows = df.close.isna()
        if not na_rows.any():
            return df

        first_session = calendar.minute_to_session(df.index[0], "next")
        last_session = calendar.minute_to_session(df.index[-1], "previous")
        opens = calendar.opens.loc[first_session:last_session]
        closes = calendar.closes.loc[first_session:last_session]

        grouper: pd.Series | pd.Grouper
        if (opens.dt.day != closes.dt.day).any():
            grouper = pd.Series(index=df.index, dtype="int32")
            for i, (open_, close) in enumerate(zip(opens, closes)):
                grouper[(grouper.index >= open_) & (grouper.index < close)] = i
        else:
            # shortcut
            grouper = pd.Grouper(freq="D")

        close_groupby = df["close"].groupby(by=grouper)
        adj_close = close_groupby.ffill()
        bv = adj_close.isna()
        if bv.any():
            open_groupby = df["open"].groupby(by=grouper)
            # bfill open to fill any missing initial rows for a session with
            # session's first registered price
            adj_close[bv] = open_groupby.bfill()[bv]

        if adj_close.isna().any():
            # at least one whole session missing prices
            bv = adj_close.isna()
            missing_sessions_groupby = adj_close[bv].groupby(by=grouper)
            # assert all values for missing sessions are missing
            assert missing_sessions_groupby.value_counts().empty
            group_sizes = missing_sessions_groupby.size()
            bv_sizes = group_sizes != 0  # pylint: disable=compare-to-zero
            sessions = group_sizes[bv_sizes].index
            if isinstance(sessions[0], (float, int)):
                sessions = pd.DatetimeIndex([opens.index[int(i)] for i in sessions])
            # fill forwards in first instance
            adj_close = adj_close.ffill()
            if adj_close.isna().any():
                # Values missing for at least one session at start of df, fill backwards
                bv = adj_close.isna()
                adj_close[bv] = df.open.bfill()[bv]
            warnings.warn(errors.PricesMissingWarning(symbol, bi, sessions, "Yahoo"))

        df.loc[:, "close"] = adj_close
        for col in ["open", "high", "low"]:
            df.loc[na_rows, [col]] = adj_close[na_rows]

        df["volume"] = df["volume"].fillna(value=0)
        return df

    def _fill_reindexed_daily(
        self,
        df: pd.DataFrame,
        cal: xcals.ExchangeCalendar,
        symbol: str,
    ) -> pd.DataFrame:
        """Fill, for a single `symbol`, rows with missing data."""
        na_rows = df.close.isna() & (df.index > self._first_trade_dates[symbol])
        if not na_rows.any():
            return df

        delay = self.delays[symbol]
        if na_rows[-1] and helpers.now() <= cal.session_open(df.index[-1]) + delay:
            na_rows.iloc[-1] = False
            if not na_rows.any():
                return df

        # fill
        adj_close = df["close"].ffill()
        bv = adj_close.isna()
        if bv.any():
            # bfill open to fill any missing initial rows with next available
            # session's open
            adj_close[bv] = df["open"].bfill()[bv]

        df.loc[na_rows, ["open", "high", "low", "close"]] = adj_close[na_rows]
        df.loc[na_rows, "volume"] = 0
        warnings.warn(
            errors.PricesMissingWarning(
                symbol, self.bis.D1, na_rows.index[na_rows], "Yahoo"
            )
        )
        return df

    @staticmethod
    def _adjust_high_low(df) -> pd.DataFrame:
        """Adjust data so that ohlc values are congruent.

        Assumes close values over incongruent high or low.
        Assumes high or low values over incongruent open.

        For any row:
            If 'close' higher than 'high', sets 'high' to 'close'.
            If 'close' lower than 'low', sets 'low' to 'close'.
            If 'open' higher than 'high', sets 'open' to 'high'.
            If 'open' lower than 'low', sets 'open' to 'low'.

        Notes
        -----
        Yahoo API has a nasty bug where occassionally the open price of the
        most recent day takes the same value as the prior day (as of
        06/11/20 only noticed issue on daily data). However, the high
        and low values will register the actual high or low of the day.
        Consequently it's possible for the day high to otherwise be lower
        than the open (observed) or the day low to be higher than the open
        (assumed). This in turn causes bqplot OHLC plots to fail (as the
        corresponding bar can not be created.)
        """
        bv = df.open > df.high
        if bv.any():
            df.loc[bv, "open"] = df.loc[bv, "high"]

        bv = df.open < df.low
        if bv.any():
            df.loc[bv, "open"] = df.loc[bv, "low"]

        bv = df.close > df.high
        if bv.any():
            df.loc[bv, "high"] = df.loc[bv, "close"]

        bv = df.close < df.low
        if bv.any():
            df.loc[bv, "low"] = df.loc[bv, "close"]

        return df

    @staticmethod
    def _remove_yahoo_duplicates(df: DataFrame, symbol: str) -> pd.DataFrame:
        if df.index.has_duplicates:
            bv = df.index.duplicated(keep="first")
            duplicates = df[bv]
            warnings.warn(errors.DuplicateIndexWarning(duplicates, symbol))
            return df[~bv]
        else:
            return df

    def _resolve_current_ts_daily(self, symbol: str, df: DataFrame) -> pd.DataFrame:
        """Set current ts to its session value."""
        last_ts = df.index[-1]
        if last_ts != last_ts.normalize():
            cal = self.calendars[symbol]
            session = cal.minute_to_session(last_ts, "previous")
            if len(df) == 1:
                df.index = pd.DatetimeIndex([session])
            else:
                df.index = df.index[:-1].insert(len(df.index) - 1, session)
        if len(df.index) > 1 and df.index[-1] == df.index[-2]:
            # This clause requried as for a time following a session close the
            # yahoo API represents the final session with two timestamps - one
            # stamped as the session open (the normal way in which the API stamps
            # a session and another as the 'live indice' stamped with, presumedly,
            # the time of the last trade. By the time it gets to here both these
            # indices are stamped as the actual session. This line removes the
            # 'live indice'.
            df = df.iloc[:-1]
        return df

    def _resolve_current_ts_intraday(
        self, symbol: str, df: DataFrame, interval: intervals.BI
    ) -> pd.DataFrame:
        """Drop last indice if a 'current timestamp' (not 'on interval')."""
        last_ts = df.index[-1]
        cal = self.calendars[symbol]
        last_session = cal.minute_to_session(last_ts)
        open_ = cal.session_open(last_session)
        remainder = (last_ts - open_) % interval
        if remainder > pd.Timedelta(0):
            df.drop(index=last_ts, inplace=True)
        return df

    def _resolve_current_ts(
        self, symbol: str, df: DataFrame, interval: intervals.BI
    ) -> pd.DataFrame:
        if interval.is_daily:
            return self._resolve_current_ts_daily(symbol, df)
        else:
            return self._resolve_current_ts_intraday(symbol, df, interval)

    def _tidy_yahoo(
        self,
        df: DataFrame,
        interval: intervals.BI,
        start: pd.Timestamp | None,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Tidy DataFrame of prices returned by `_request_yahoo--`."""
        # pylint: disable=too-complex, too-many-locals, too-many-branches
        # pylint: disable=too-many-statements
        df = helpers.order_cols(df)
        groupby = df.groupby(level="symbol")
        sdfs, empty_sdfs = [], []
        for symbol in self.symbols:

            def get_columns_index(index: pd.Index | None = None) -> pd.MultiIndex:
                if index is None:
                    index = pd.Index(helpers.AGG_FUNCS.keys())
                parts = [[symbol], index]  # pylint: disable=cell-var-from-loop
                return pd.MultiIndex.from_product(parts, names=["symbol", ""])

            # take copy of group to avoid later setting to a slice of a DataFrame.
            try:
                sdf = groupby.get_group(symbol).copy()
            except KeyError:
                if not self.has_single_calendar:
                    # no prices for symbol over requested dr
                    empty_sdfs.append(pd.DataFrame(columns=get_columns_index()))
                    continue
                raise

            sdf = sdf.droplevel("symbol")
            if interval.is_daily:
                sdf.index = pd.DatetimeIndex(sdf.index)
                sdf = self._adjust_high_low(sdf)
            sdf = self._resolve_current_ts(symbol, sdf, interval)
            if sdf.empty:
                # This can happen if the only indice of sdf was the live indice.
                # Particular case is bug that on the yahoo API side where can (albeit
                # seemingly rare) temporarily fail to return prices for a specific day.
                # If prices requested for only that day or a period of then will have
                # reached here will live indice only.
                empty_sdfs.append(pd.DataFrame(columns=get_columns_index()))
                continue
            sdf = self._remove_yahoo_duplicates(sdf, symbol)
            start = start if start is not None else sdf.index[0]
            calendar = self.calendars[symbol]
            index = self._get_trading_index(calendar, interval, start, end)
            reindex_index = (
                index if interval.is_daily else index.left  # type: ignore[union-attr]
            )
            sdf = sdf.reindex(reindex_index)
            if interval.is_intraday:
                sdf = self._fill_reindexed(sdf, calendar, interval, symbol)
            else:
                sdf = self._fill_reindexed_daily(sdf, calendar, symbol)
            sdf.index = index
            sdf.columns = get_columns_index(sdf.columns)
            if sdf.empty:
                empty_sdfs.append(sdf)
            else:
                sdfs.append(sdf)

        if not sdfs:
            # Can happen if prices are returned by yahoo although all returned
            # prices fall outside of the expected trading index. An example is prices
            # for, at least, hong kong stocks include a single index after the close.
            params = dict(interval=interval, start=start, end=end)
            raise errors.PricesUnavailableFromSourceError(params, df)
        df = pd.concat(sdfs, axis=1)

        if empty_sdfs:
            # add symbols for which prices unavailable over requested period
            columns = df.columns
            for empty_sdf in empty_sdfs:
                columns = columns.union(empty_sdf.columns)
            df = df.reindex(columns=columns)

        df.sort_index(inplace=True)
        df.columns = df.columns.set_names("symbol", level=0)
        return df

    # Abstract base methods.

    def _request_data_1m(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        """Request data for '1m' interval."""
        start_, end_ = start, end
        interval = self.BaseInterval.T1
        MAX_DAYS_PER_REQUEST = pd.Timedelta(6, "D")  # 1 day margin

        # evalute max days from the prior indice that the volume glitch
        # fix (to `_request_data`) will set start to.
        prior_indice = self._adj_start_for_vol_glitch(start, interval)
        end_ = min(end, prior_indice + MAX_DAYS_PER_REQUEST)
        df = self._request_yahoo(interval=interval, start=start, end=end_)
        next_df = pd.DataFrame()
        while True:
            df = pd.concat([df, next_df])
            df = df.drop_duplicates()  # only occurs for some symbols, on the join
            start = end_
            prior_indice = self._adj_start_for_vol_glitch(start, interval)
            end_ = min(end, prior_indice + MAX_DAYS_PER_REQUEST)
            if end_ == start:
                break
            try:
                next_df = self._request_yahoo(interval=interval, start=start, end=end_)
            except errors.PricesUnavailableFromSourceError:
                break
        return df, start_, end_

    def _request_data(
        self,
        interval: intervals.BI,
        start: pd.Timestamp | None,
        end: pd.Timestamp,
    ) -> DataFrame:
        if start is None and interval.is_intraday:
            raise ValueError(
                "`start` cannot be None if `interval` is intraday. `interval`"
                f"receieved as 'f{interval}'."
            )
        if interval == self.BaseInterval.T1:
            assert start is not None
            prices, start, end = self._request_data_1m(start, end)
        else:
            end_ = end
            if interval.is_daily:
                # 22 hours ensures markets opening in Americas included
                # whilst avoiding including the following session of
                # Australasian markets
                end_ += pd.Timedelta(22, "H")
            prices = self._request_yahoo(interval=interval, start=start, end=end_)
        return self._tidy_yahoo(prices, interval, start, end)

    @staticmethod
    def _remove_non_trading_indices(
        df: pd.DataFrame, cals: list[xcals.ExchangeCalendar]
    ) -> pd.DataFrame:
        """Remove indices that include no minutes of any of `cals`."""
        non_trading = df.pt.indices_non_trading(cals[0])
        for cal in cals[1:]:
            non_trading = non_trading.intersection(df.pt.indices_non_trading(cal))
        return df.drop(labels=non_trading)

    def prices_for_symbols(self, symbols: Symbols) -> base.PricesBase:
        """Return PricesYahoo instance for one or more symbols.

        Populates instance with any pre-existing price data.

        Parameters
        ----------
        symbols
            Symbols to include to the new instance. Passed as class'
            'symbols' parameter.
        """
        # pylint: disable=protected-access
        symbols = helpers.symbols_to_list(symbols)
        difference = set(symbols).difference(set(self.symbols))
        if difference:
            msg = (
                "symbols must be a subset of Prices' symbols although"
                f" received the following symbols which are not:"
                f" {difference}.\nPrices symbols are {self.symbols}."
            )
            raise ValueError(msg)

        cals_all = {s: self.calendars[s] for s in symbols}
        prices_obj = type(self)(symbols=symbols, calendars=cals_all)

        cals = list(prices_obj.calendars_unique)
        fewer_cals = len(cals) < len(self.calendars_unique)
        for bi in self.bis:  # type: ignore[attr-defined]  # enum has __iter__ attr.
            new_pdata = copy.deepcopy(self._pdata[bi])
            if new_pdata._table is not None:
                table = new_pdata._table[symbols].copy()
                if fewer_cals:
                    table = self._remove_non_trading_indices(table, cals)
                new_pdata._table = table
            prices_obj._pdata[bi] = new_pdata
        return prices_obj
