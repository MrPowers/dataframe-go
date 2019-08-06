package forecast

import (
	"context"
	"errors"

	"github.com/rocketlaunchr/dataframe-go"
)

// SimpleExponentialSmoothing method calculates
// and returns forecast for future m periods
//
//// s - dataframe.SeriesFloat64 object
//
// y - Time series data gotten from s.
// alpha - Exponential smoothing coefficients for level, trend,
//        seasonal components.
// m - Intervals into the future to forecast
//
// https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc431.htm
// newvalue = smoothing * next + (1 - smoothing)*old value
// forecast[i+1] = St[i] + alpha * ϵt,
// where ϵt is the forecast error (actual - forecast) for period i.
func SimpleExponentialSmoothing(ctx context.Context, s *dataframe.SeriesFloat64, alpha float64, m int, r ...dataframe.Range) (*dataframe.SeriesFloat64, error) {

	if len(r) == 0 {
		r = append(r, dataframe.Range{})
	}

	start, end, err := r[0].Limits(len(s.Values))
	if err != nil {
		return nil, err
	}
	// inclusive of value at end index
	y := s.Values[start : end+1]

	// Validating arguments
	if len(y) == 0 {
		return nil, errors.New("value of y should be not null")
	}

	if m <= 0 {
		return nil, errors.New("value of m must be greater than 0")
	}

	if m > len(y) {
		return nil, errors.New("value of m can not be greater than length of y")
	}

	if (alpha < 0.0) || (alpha > 1.0) {
		return nil, errors.New("value of Alpha should satisfy 0.0 <= alpha <= 1.0")
	}

	st := make([]float64, len(y))
	forecast := make([]float64, len(y)+m)

	// Set initial value to first element in y
	st[1] = y[0]

	// start smoothing from the third element
	for i := 2; i < len(y); i++ {

		// Exiting on context error
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// simple exponential Smoothing
		st[i] = alpha*y[i] + ((1.0 - alpha) * st[i-1])

		// forecast
		if (i + m) >= len(y) {
			forecast[i+m] = st[i] + (alpha * (y[i] - st[i]))
		}
	}

	init := &dataframe.SeriesInit{}

	seriesForecast := dataframe.NewSeriesFloat64("Forecast", init)

	// Load forecast data into series
	seriesForecast.Insert(seriesForecast.NRows(), forecast[len(y):])

	return seriesForecast, nil

}
