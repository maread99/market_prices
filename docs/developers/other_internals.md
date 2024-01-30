# Internals

## Intervals not a factor of session length

If all of...
* prices are anchored to a session open
* the period ends on the session close
* the interval is not a factor of the session  

...then the final indice will not align with the session close, rather the left side of the final indice falls before the session close and the right side falls afterwards.

Example, 1 hour base interval for a NYSE session that opens 09.30 and closes 16.00 (local). The data will have an unaligned final indice 15.30 - 16.30.

Where should the period end be set to? To the start of the unaligned indice, its end, or the session close? Most important is where the end should NOT be set to. In accordance with the Golder Rule (see periods.ipynb tutorial) the end should NEVER be set to the right of the unaligned indice (i.e. to after the session close) IF any symbol trades between the session close and the right of the unaligned indice. To the contrary, setting the end to the right of the unaligned indice would result in introducing prices registered after the evaluated period end - a big no.

### So, where should the end be set?

* If any symbol trades between the session close and the right of the unaligned indice, the end should be set to the left of the unaligned indice. This means the period from the left of the unaligned indice to the session close will not be represented in any prices requested to a session end.

* If no symbol trades between the session close and the right of the unaligned indice:
    * If the end is to represent the datetime to which prices should be requested then it should be set to the right of the unaligned indice. This allows for the requested prices to cover the period through to the session end and nothing beyond.

    * If the end is to ascertain accuracy (i.e. how close to a requested datetime that the base interval can serve prices) then end should be set to the session close, i.e. the datetime to which it is accurate.

### How internals accommodate session end unaligned indices

Session close unaligned indices are not a consideration when prices are anchored to 'workback'. (Only fully aligned base intervals are used to evaluate prices anchored 'workback' - the tables of such base intervals fully comprise of trading indices, i.e. at least one symbol will be trading at every time covered by every indice.)

`daterange.GetterIntraday` evaluates unaligned session closes against indices based on its `end_alignment_interval`. This in turn is determined by its `end_alignment` constructor argument. `end_alignment` can take either `Alignment.BI` or `Alignment.FINAL` to determine if indices should be defined against the base interval (`interval`) or the final interval (`final_interval`).

`daterange.GetterIntraday` sets the range end such that if the range ends on a session close (evaluated against `calendar`) that does not align with indices (evaluated against `end_alignment_interval`) then range end is set:
* as the left of the final indice (i.e. right of prior indice) if any symbol is open during the part of that final indice that falls after the session close, i.e. the otherwise final indice is excluded.
* as the right of the final indice if no symbol is open during the part of the final indice that falls after the session close.

This allows for prices to be requested for a range that will ensure the session end is included if it can be - sources are unlikely to return the final indice unless the requested range fully covers the unaligned indice. Using the '1h' interval NYSE example above, if prices were requested from Yahoo through to the 16.00 session end the returned data would end at 15.30 (indexed as 14.30 to reflect the period 14.30 through 15.30). It would not include data for the 15.30 - 16.30 indice as this indice was not fully covered by date range requested. (In the same way, `PricesBase._get_table_part` will only return indices of base tables that are fully contained within the passed daterange.)

To express range end **accuracy** `daterange.GetterIntraday.daterange` and `daterange.GetterIntraday.get_end()` return a tuple, the second item of which gives the accuracy of the range end indicated by the first item. This allows for the accuracy of the range end to be interrogated when evaluating which base intervals can best serve a call.

The `PricesBase.get()` method provides the **'openend'** argument so the client can determine how to treat any unaligned end indice:  

    openend : Literal["maintain", "shorten"], default: "maintain"
        Only relevant if anchor is 'open', otherwise ignored.

        Determines how the final indice should be defined if `end`
        evaluates to a session close (as evaluated against
        `lead_symbol` calendar) which does not align with the indices:

            "maintain" (default) (maintain the interval):
                The final indice will have length `interval`.

                Considering the period between the session close and
                the right of the indice that contains the session
                close:
                    If no symbol trades during this period then the
                    final indice will be the indice that contains the
                    session close, such that the right side of the
                    final indice will be defined to the right of the
                    session close.

                    If any symbol trades during this period then the
                    final indice will be the latest indice with a
                    right side that preceeds the session close.
                    Consequently the right of the final indice will be
                    to the left of the session close and prices at the
                    session close will not be included.

                Note: The final indice may still be shortened if
                `force` is True.

            "shorten":
                Define final indice shorter than `interval` in order
                that right side of final indice aligns with session
                close.

                The final interval will only be defined in this way
                if either:
                    No symbol trades during the part of the final
                    indice that falls after the session close.
                    
                    Data is available to create the table by
                    downsampling data of a base interval that aligns
                    with the session close.

                If it is not possible to define a shorter indice then
                the final indice will be defined as for "maintain".

        NOTE In no event will the final indice include prices
        registered after the evaluated period end.

        See anchor.ipynb tutorial for further explanation and examples.

    openend : Literal['maintain', 'shorten'], default: 'maintain'
        How should the final indice be defined if period end is
        a session close (as evaluated against 'lead_symbol')
        and indices do not align with that close.
            'maintain' - maintain the interval.
                If no symbol trades during the part of the
                final interval that falls after the session
                close then the right side of the final indice
                will be defined to the right of the session
                close.

                If any symbol trades during the part of the
                final interval that falls after the session
                close then the right side of the final indice
                will be defined to the left of the session
                close (such that the prices at the end of the
                session will not be included).

            'shorten' - shorten the final interval such that
            the right side of the final indice falls on
            the session end, i.e. the final indice will be
            shorter than the regular interval. The final
            interval will only be shortened if either:
                No symbol trades during the part of the final
                interval that falls after the session close.

                Data is available to create the table by
                downsampling data of a base interval that
                aligns with the session close.

            If it is not possible to shorten the final indice
            then the final indice will defined as for
            'maintain'.

Internally this is handled by:
* Setting `daterange.GetterIntraday.end_alignment` to Alignment.BI if 'openend' is "shorten" or to Alignment.FINAL if 'openend' is 'maintain' (NB 'openend' is always set to Alignment.BI if 'open' is "workback").
* `daterange` will then evalute the unaligned session close.
    * If 'openend' is "maintain" then `daterange` will evaluate unaligned session end indices based on the final interval (i.e. downsample interval or base interval if there is no downsample interval).

    * If 'openend' is "shorten" then `daterange` will evaluate unaligned session end indices based on the base inteval. If the range ends on a session close which is aligned against the base interval, but not against the downsample interval, this will result in prices being included, at the base interval, from the end of what would be the final downsample indice (that ends before the session close) through to the session close. These indices collectively represent a partial trading indice when evaluated against the downsample interval (the partial indice would extend beyond the session close). If 'openend' is "shorten" `PricesBase._get_table_intraday()` then checks if the right of the last indice of the downsampled prices is later than the right of the last indice of the base interval prices from which the downsample prices were evaluated. If it is, this indicates the final indice was aggregated from insufficent base indices to fully reflect the downsampled indice. In this case the final downsampled indice is shortened so that its right side reflects the right of the last indice of the base interval data (i.e. the unalgined session close).

NB: A daterange end that is a session close that is aligned against the base interval although not against the downsample interval is the only circumstance that result in the base table used by `PricesData._get_table_intraday()` having a number of indices that are not a multiple of the downsample factor (downsample interval / base interval). In all other cases the daterange is evaluted against indices based on the final interval and hence there is no remainder of base interval indices that would otherwise contribute to an incomplete downsampled indice.

## `daterange.GetterIntraday.daterange_tight`
The `daterange.GetterIntraday.daterange_tight` tightens the end (first item) so that the difference between the end and the accuracy is less than one base interval. When the downsample interval (ds_interval) is high relative to the base interval this has the benefit of preventing 'partial requests' for data that cover just the end of the daterange over which no symbol trades (and hence no data would be expected to be returned). See the property documentation for an example.

## Right Limits
Right limits were introduced January 2024 to support sourcing historic price data from locally stored .csv files (`PricesCsv`). To minimise changes and maintain much of the existing implementation (and hence tests and tutorials) a new `BASE_LIMITS_RIGHT` class attribute was introduced rather than altering the existing `BASE_LIMITS` which could have been amended to take a tuple defining both the left and right limit. Similarly, a 'limit_right' parameter was added to the `daterange._Getter` constructor rather than redefining the 'limit' parameter as 'limits' to take a tuple. On reflection, given the different treatment of each limit (with the right limit defaulting to 'now' if not defined, whereas the left limit for daily data has to be found) I think that defining left and right limits separately IS definitely the way to do. This way the right limit can be ignored for any service that provides for data to now (i.e. most of them). It's much clearer to have them left and right limits separately. Keep it this way.
