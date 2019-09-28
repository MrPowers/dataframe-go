package forecast

import (
	"context"
	"errors"
	"fmt"

	"github.com/bradfitz/iter"
	"github.com/rocketlaunchr/dataframe-go"
)

// SesModel is a struct that holds necessary
// computed values for a forecasting result
type SesModel struct {
	testData       *dataframe.SeriesFloat64
	fcastData      *dataframe.SeriesFloat64
	initialLevel   float64
	originValue    float64
	smoothingLevel float64
	alpha          float64
	mae            float64
	sse            float64
	rmse           float64
	mape           float64
}

// SimpleExponentialSmoothing performs forecasting based on the Exponential Smoothing algorithm.
// It returns a SeriesFloat64 with the next m forecasted values in the series.
// The argument α must be between [0,1]. Recent values receive more weight when α is closer to 1.
func SimpleExponentialSmoothing(ctx context.Context, s *dataframe.SeriesFloat64, α float64, r ...dataframe.Range) (*SesModel, error) {

	if len(r) == 0 {
		r = append(r, dataframe.Range{})
	}

	count := len(s.Values)
	if count == 0 {
		return nil, errors.New("no values in series range")
	}

	start, end, err := r[0].Limits(count)
	if err != nil {
		return nil, err
	}

	// Validation
	if end-start < 1 {
		return nil, errors.New("no values in series range")
	}

	if (α < 0.0) || (α > 1.0) {
		return nil, errors.New("α must be between [0,1]")
	}

	trainedModel := &SesModel{
		alpha:          α,
		testData:       &dataframe.SeriesFloat64{},
		fcastData:      &dataframe.SeriesFloat64{},
		initialLevel:   0.0,
		smoothingLevel: 0.0,
		mae:            0.0,
		sse:            0.0,
		rmse:           0.0,
		mape:           0.0,
	}

	testData := s.Values[end+1:]
	if len(testData) < 2 {
		return nil, errors.New("There should be a minimum of 2 data left as testing data")
	}

	testSeries := dataframe.NewSeriesFloat64("Test Data", nil)
	testSeries.Values = testData

	trainedModel.testData = testSeries

	// tCount := len(testData)

	var st float64

	for i := start; i < end+1; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		xt := s.Values[i]

		if i == start {
			st = xt
			trainedModel.initialLevel = xt
		} else {
			st = α*xt + (1-α)*st
		}

	}

	fcast := []float64{}
	for k := end + 1; k < len(s.Values); k++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		xt := s.Values[k]

		st = α*xt + (1-α)*st
		fcast = append(fcast, st)

		// Setting the last value in data as Yorigin value for bootstrapping
		if k == len(s.Values)-1 {
			trainedModel.originValue = s.Values[k]
		}
	}

	fcastSeries := dataframe.NewSeriesFloat64("Forecast Data", nil)
	fcastSeries.Values = fcast
	trainedModel.fcastData = fcastSeries

	trainedModel.smoothingLevel = st

	opts := &ErrorOptions{}

	mae, _, err := MeanAbsoluteError(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	sse, _, err := SumOfSquaredErrors(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	rmse, _, err := RootMeanSquaredError(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	mape, _, err := MeanAbsolutePercentageError(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	trainedModel.sse = sse
	trainedModel.mae = mae
	trainedModel.rmse = rmse
	trainedModel.mape = mape

	return trainedModel, nil
}

// Predict method is used to run future predictions for Ses
// Using Bootstrapping method
func (sm *SesModel) Predict(ctx context.Context, m int) (*dataframe.SeriesFloat64, error) {
	if m <= 0 {
		return nil, errors.New("m must be greater than 0")
	}

	forecast := make([]float64, 0, m)
	α := sm.alpha
	Yorigin := sm.originValue
	st := sm.smoothingLevel

	// Now calculate forecast
	for range iter.N(m) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		st = α*Yorigin + (1-α)*st
		forecast = append(forecast, st)
	}

	fdf := dataframe.NewSeriesFloat64("Prediction", nil)
	fdf.Values = forecast

	return fdf, nil
}

// Summary function is used to Print out Data Summary
// From the Trained Model
func (sm *SesModel) Summary() {

	fmt.Println(sm.testData.Table())
	fmt.Println(sm.fcastData.Table())

	alpha := dataframe.NewSeriesFloat64("Alpha", nil, sm.alpha)
	initLevel := dataframe.NewSeriesFloat64("Initial Level", nil, sm.initialLevel)
	st := dataframe.NewSeriesFloat64("Smooting Level", nil, sm.smoothingLevel)

	info := dataframe.NewDataFrame(alpha, initLevel, st)
	fmt.Println(info.Table())

	mae := dataframe.NewSeriesFloat64("MAE", nil, sm.mae)
	sse := dataframe.NewSeriesFloat64("SSE", nil, sm.sse)
	rmse := dataframe.NewSeriesFloat64("RMSE", nil, sm.rmse)
	mape := dataframe.NewSeriesFloat64("MAPE", nil, sm.mape)
	accuracyErrors := dataframe.NewDataFrame(sse, mae, rmse, mape)

	fmt.Println(accuracyErrors.Table())
}

// Optimize method tunes the model result and tries to reduce
// Accuracy Errors To the mininum
func (sm *SesModel) Optimize() (*SesModel, error) {
	// To do.
	panic("Model Optimize Tuner To be implemented soon")
}
